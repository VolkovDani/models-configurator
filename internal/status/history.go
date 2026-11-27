package status

import (
	_ "embed"
	"encoding/json"
	"html/template"
	"io"
	"net/http"
	"strconv"

	"github.com/VolkovDani/models-configurator/internal/history"
	"github.com/VolkovDani/models-configurator/internal/manager"
)

//go:embed history.html
var historyHTML string
var historyPage = newPanelTemplate("history", historyHTML, template.FuncMap{"endpoint": manager.InferenceEndpoint})

//go:embed history.js
var historyScript string

//go:embed chart/history-chart.js
var historyChartScript string

//go:embed chart/licenses.txt
var historyChartLicenses string

const historyPageSize = 50

func RegisterHistory(mux *http.ServeMux, store *history.Store) {
	handler := historyHandler{store: store}
	mux.Handle("/history", handler)
	mux.Handle("/history.json", handler)
	mux.Handle("/history.js", handler)
	mux.Handle("/history-chart.js", handler)
	mux.Handle("/history-chart-licenses.txt", handler)
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
	switch r.URL.Path {
	case "/history.js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		io.WriteString(w, historyScript)
		return
	case "/history-chart.js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		io.WriteString(w, historyChartScript)
		return
	case "/history-chart-licenses.txt":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.WriteString(w, historyChartLicenses)
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
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
	w.Header().Set("Referrer-Policy", "no-referrer")
	historyPage.Execute(w, historyView{locale: words, Page: page, OlderPage: before != 0})
}
