// Package status reports provider metadata for explicitly configured models.
package status

import (
	"context"
	_ "embed"
	"encoding/json"
	"html/template"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

//go:embed panel.html
var htmlSource string
var page = template.Must(template.New("panel").Parse(htmlSource))

type row struct {
	Name     string `json:"name"`
	ID       string `json:"id"`
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Role     string `json:"role"`
	State
}

type snapshot struct {
	CheckedAt time.Time `json:"checked_at"`
	Models    []row     `json:"models"`
}

type flight struct {
	done   chan struct{}
	result snapshot
}

type panel struct {
	cfg     manager.Config
	mu      sync.Mutex
	pending *flight
}

func New(cfg manager.Config) http.Handler {
	return &panel{cfg: cfg}
}

func (p *panel) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/status" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	p.mu.Lock()
	f := p.pending
	if f == nil {
		f = &flight{done: make(chan struct{})}
		p.pending = f
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
			defer cancel()
			f.result = collect(ctx, p.cfg)
			p.mu.Lock()
			p.pending = nil
			close(f.done)
			p.mu.Unlock()
		}()
	}
	p.mu.Unlock()
	select {
	case <-r.Context().Done():
		return
	case <-f.done:
	}
	if r.URL.Path == "/status" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(f.result)
		return
	}
	lang := r.URL.Query().Get("lang")
	if lang != "ru" {
		lang = "en"
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Language", lang)
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
	w.Header().Set("Referrer-Policy", "no-referrer")
	page.Execute(w, view{snapshot: f.result, Lang: lang, Words: translations[lang]})
}

func collect(ctx context.Context, cfg manager.Config) snapshot {
	rows := make([]row, len(cfg.Models))
	groups := make(map[endpoint][]int)
	for j, m := range cfg.Models {
		e := endpoint{provider: m.Provider, base: strings.TrimRight(m.BaseURL, "/"), keyEnv: m.KeyEnv}
		groups[e] = append(groups[e], j)
		rows[j] = row{Name: m.Name, ID: m.ID(), Model: m.Model, Provider: m.Provider, Role: m.Role}
	}
	jobs := make(chan endpoint, len(groups))
	for e := range groups {
		jobs <- e
	}
	close(jobs)
	var wg sync.WaitGroup
	for range min(4, len(groups)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for e := range jobs {
				result := inventory{provider: e.provider, reachability: "unchecked"}
				if ctx.Err() == nil {
					result = probe(ctx, e)
				}
				for _, j := range groups[e] {
					m := cfg.Models[j]
					rows[j].State = result.state(m.Model, m.Digest)
				}
			}
		}()
	}
	wg.Wait()
	return snapshot{CheckedAt: time.Now().UTC(), Models: rows}
}

type view struct {
	snapshot
	Lang  string
	Words map[string]string
}

func (v view) Label(code string) string { return v.Words[code] }
func (v view) Flag(value *bool) string {
	if value == nil {
		return v.Label("unknown")
	}
	if *value {
		return v.Label("yes")
	}
	return v.Label("no")
}
func (v view) Timestamp() string { return v.CheckedAt.Format(time.RFC3339) }
