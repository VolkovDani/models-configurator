package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestAPIDocumentation(t *testing.T) {
	t.Setenv("MODELS_MAX_PARALLEL_REQUESTS", "1")
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "history.jsonl"))
	h, err := service(manager.Config{Models: []manager.Model{}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		path, contentType, text string
	}{
		{"/", "text/html", "href=\"/docs/\""},
		{"/docs/", "text/html", "src=\"./swagger-initializer.js\""},
		{"/docs/", "text/html", "src=\"./swagger-ui-bundle.js\""},
		{"/docs/", "text/html", "href=\"./swagger-ui.css\""},
		{"/docs/swagger-initializer.js", "text/javascript", "/openapi.json"},
		{"/docs/swagger-ui-bundle.js", "text/javascript", "SwaggerUIBundle"},
		{"/docs/swagger-ui.css", "text/css", ".swagger-ui"},
	} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "http://localhost"+tc.path, nil))
		if w.Code != http.StatusOK || !strings.HasPrefix(w.Header().Get("Content-Type"), tc.contentType) || !strings.Contains(w.Body.String(), tc.text) {
			t.Errorf("GET %s: status %d, Content-Type %q, want %q in body", tc.path, w.Code, w.Header().Get("Content-Type"), tc.text)
		}
	}
	for _, path := range []string{"/docs/", "/openapi.json"} {
		r := httptest.NewRequest(http.MethodGet, "http://localhost"+path, nil)
		r.Header.Set("Origin", "http://other.example")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != http.StatusForbidden {
			t.Errorf("cross-origin GET %s: status %d, want 403", path, w.Code)
		}
	}
}

func TestOpenAPIOperations(t *testing.T) {
	t.Setenv("MODELS_MAX_PARALLEL_REQUESTS", "1")
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "history.jsonl"))
	h, err := service(manager.Config{Models: []manager.Model{}})
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "http://localhost/openapi.json", nil))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("GET /openapi.json: status %d, Content-Type %q", w.Code, w.Header().Get("Content-Type"))
	}
	var spec struct {
		OpenAPI string
		Paths   map[string]map[string]json.RawMessage
	}
	if err := json.Unmarshal(w.Body.Bytes(), &spec); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(spec.OpenAPI, "3.1.") {
		t.Fatalf("OpenAPI version %q, want 3.1.x", spec.OpenAPI)
	}
	getPaths := []string{"/status", "/v1/models"}
	postPaths := []string{"/v1/chat/completions", "/v1/chat/token-count", "/v1/embeddings", "/v1/decisions", "/v1/bge-m3/encode"}
	if len(spec.Paths) != len(getPaths)+len(postPaths) {
		t.Errorf("documented %d paths, want %d", len(spec.Paths), len(getPaths)+len(postPaths))
	}
	for _, tc := range []struct {
		method string
		paths  []string
		status int
	}{
		{http.MethodGet, getPaths, http.StatusOK},
		{http.MethodPost, postPaths, http.StatusBadRequest},
	} {
		for _, path := range tc.paths {
			if _, found := spec.Paths[path][strings.ToLower(tc.method)]; !found {
				t.Errorf("OpenAPI is missing %s %s", tc.method, path)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest(tc.method, "http://localhost"+path, strings.NewReader("{}")))
			if w.Code != tc.status {
				t.Errorf("%s %s: status %d, want %d", tc.method, path, w.Code, tc.status)
			}
		}
	}
}
