package main

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestWebPanelTheme(t *testing.T) {
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "history.jsonl"))
	handler, err := service(manager.Config{Models: []manager.Model{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, route := range []string{"/", "/settings", "/history"} {
		t.Run(route, func(t *testing.T) {
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "http://localhost"+route+"?lang=ru", nil))
			if w.Code != http.StatusOK {
				t.Fatalf("status %d", w.Code)
			}
			for _, fragment := range []string{`href="/theme.css"`, `src="/theme.js"`, `id="panel-theme"`, `<label for="panel-theme">Тема</label>`, "Системная", "Светлая", "Тёмная"} {
				if !strings.Contains(w.Body.String(), fragment) {
					t.Errorf("missing %q", fragment)
				}
			}
			policy := w.Header().Get("Content-Security-Policy")
			for _, directive := range []string{"style-src 'self'", "script-src 'self'"} {
				if !strings.Contains(policy, directive) {
					t.Errorf("policy %q is missing %q", policy, directive)
				}
			}
		})
	}
	for _, asset := range []struct{ route, media string }{
		{"/theme.css", "text/css; charset=utf-8"},
		{"/theme.js", "text/javascript; charset=utf-8"},
	} {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "http://localhost"+asset.route, nil))
		if w.Code != http.StatusOK || w.Header().Get("Content-Type") != asset.media || w.Body.Len() == 0 {
			t.Errorf("%s returned status %d and content type %q", asset.route, w.Code, w.Header().Get("Content-Type"))
		}
	}
}
