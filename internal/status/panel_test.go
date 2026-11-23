package status

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestPanelAndJSON(t *testing.T) {
	var calls atomic.Int64
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		fmt.Fprint(w, `{"data":[{"id":"first"},{"id":"second"},{"id":"PRIVATE_unconfigured"}]}`)
	}))
	defer up.Close()
	cfg := manager.Config{Models: []manager.Model{
		{Name: "alpha", Provider: "openai-compatible", BaseURL: up.URL + "/v1", Role: "chat", Model: "first", Revision: "v1"},
		{Name: "beta", Provider: "openai-compatible", BaseURL: up.URL + "/v1", Role: "embeddings", Model: "second", Revision: "v1"},
	}}
	h := New(cfg)
	for _, tc := range []struct {
		method, path, text string
		code               int
	}{
		{"GET", "/status", "\"checked_at\"", 200},
		{"GET", "/?lang=en", "Memory placement", 200},
		{"GET", "/?lang=ru", "Размещение в памяти", 200},
		{"POST", "/status", "", 405},
		{"GET", "/missing", "", 404},
	} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, nil))
		body := w.Body.String()
		if w.Code != tc.code || !strings.Contains(body, tc.text) || strings.Contains(body, "PRIVATE") || strings.Contains(body, up.URL) {
			t.Errorf("%s: code %d, body %s", tc.path, w.Code, body)
		}
		if tc.path == "/status" && tc.code == 200 {
			var snap struct {
				CheckedAt time.Time `json:"checked_at"`
				Models    []struct {
					ID, Role string
					Listed   *bool
				}
			}
			if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil || snap.CheckedAt.IsZero() || len(snap.Models) != 2 || snap.Models[0].ID != cfg.Models[0].ID() || snap.Models[1].Role != "embeddings" || snap.Models[0].Listed == nil || !*snap.Models[0].Listed {
				t.Fatalf("invalid snapshot: %s", body)
			}
		}
	}
	if calls.Load() != 3 {
		t.Errorf("got %d metadata requests", calls.Load())
	}
}

func TestPinnedState(t *testing.T) {
	size := int64(0)
	i := inventory{provider: "ollama", reachability: "reachable", available: []modelInfo{{Name: "tag", Digest: "abc"}}, running: []modelInfo{{Name: "tag", Digest: "def", Size: 100, VRAM: &size}}}
	s := i.state("tag", "sha256:abc")
	if s.Installed == nil || !*s.Installed || s.Loaded == nil || *s.Loaded || s.Memory != "unknown" {
		t.Fatalf("wrong pinned state %+v", s)
	}
	i.available[0].Digest = ""
	if s := i.state("tag", "sha256:abc"); s.Installed != nil {
		t.Fatal("missing digest reported as checked")
	}
}

func TestCloudAliasDoesNotProveLocalModel(t *testing.T) {
	i := inventory{provider: "ollama", available: []modelInfo{{Name: "tag", Digest: "abc", RemoteModel: "remote"}}}
	s := i.state("tag", "abc")
	if s.Installed == nil || *s.Installed {
		t.Errorf("cloud alias was reported installed: %+v", s)
	}
}

func TestTranslationsHaveSameKeys(t *testing.T) {
	for lang, labels := range translations {
		for key := range translations["en"] {
			if labels[key] == "" {
				t.Errorf("%s: %s missing", lang, key)
			}
		}
		for key := range labels {
			if translations["en"][key] == "" {
				t.Errorf("%s: extra key %s", lang, key)
			}
		}
	}
}

func TestEmptyInventory(t *testing.T) {
	h := New(manager.Config{})
	for _, tc := range []struct{ lang, message string }{{"en", "No models configured"}, {"ru", "Модели не настроены"}} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", "/?lang="+tc.lang, nil))
		if !strings.Contains(w.Body.String(), tc.message) {
			t.Errorf("missing empty state for %s", tc.lang)
		}
	}
}

// Signals when a request reaches the wait for metadata so that concurrent
// requests can be tested without time-based scheduling assumptions.
type waitingContext struct {
	context.Context
	once    sync.Once
	waiting chan struct{}
}

func (c *waitingContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}

func TestConcurrentStatusReadersShareProbe(t *testing.T) {
	var calls atomic.Int64
	started := make(chan struct{})
	release := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			close(started)
		}
		<-release
		fmt.Fprint(w, `{"data":[]}`)
	}))
	defer up.Close()
	h := New(manager.Config{Models: []manager.Model{{Name: "one", Provider: "openai-compatible", BaseURL: up.URL, Model: "m"}}})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	firstDone := make(chan struct{})
	go func() {
		h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequestWithContext(ctx, "GET", "/status", nil))
		close(firstDone)
	}()
	<-started
	cancel()
	<-firstDone
	secondCtx := &waitingContext{Context: context.Background(), waiting: make(chan struct{})}
	w := httptest.NewRecorder()
	secondDone := make(chan struct{})
	go func() {
		h.ServeHTTP(w, httptest.NewRequestWithContext(secondCtx, "GET", "/status", nil))
		close(secondDone)
	}()
	<-secondCtx.waiting
	close(release)
	<-secondDone
	if calls.Load() != 1 || w.Code != 200 || !strings.Contains(w.Body.String(), `"reachability":"reachable"`) {
		t.Errorf("calls %d, second result %d %s", calls.Load(), w.Code, w.Body)
	}
}
