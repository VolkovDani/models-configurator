package status

import (
	_ "embed"
	"encoding/json"
	"html/template"
	"io"
	"net/http"
	"strconv"

	"github.com/VolkovDani/models-configurator/internal/history"
)

//go:embed history.html
var historyHTML string
var historyPage = template.Must(template.New("history").Parse(historyHTML))

//go:embed history.js
var historyScript string

const historyPageSize = 50

func RegisterHistory(mux *http.ServeMux, store *history.Store) {
	handler := historyHandler{store: store}
	mux.Handle("/history", handler)
	mux.Handle("/history.json", handler)
	mux.Handle("/history.js", handler)
}

type historyHandler struct {
	store *history.Store
}

type historyView struct {
	locale
	history.Page
	OlderPage bool
}

func (h historyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	if r.URL.Path == "/history.js" {
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		io.WriteString(w, historyScript)
		return
	}

	words := localize(r)
	var before uint64
	if raw := r.URL.Query().Get("before"); raw != "" {
		var err error
		before, err = strconv.ParseUint(raw, 10, 64)
		if err != nil || before == 0 {
			http.Error(w, words.Label("history_cursor_error"), http.StatusBadRequest)
			return
		}
	}
	page := h.store.List(before, historyPageSize)
	if r.URL.Path == "/history.json" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(page)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Language", words.Lang)
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
	w.Header().Set("Referrer-Policy", "no-referrer")
	historyPage.Execute(w, historyView{locale: words, Page: page, OlderPage: before != 0})
}
