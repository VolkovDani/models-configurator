package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestParallelRequestLimit(t *testing.T) {
	const key = "MODELS_MAX_PARALLEL_REQUESTS"
	t.Setenv(key, "")
	if err := os.Unsetenv(key); err != nil {
		t.Fatal(err)
	}
	if got, err := parallelRequestLimit(); got != 1 || err != nil {
		t.Fatalf("unset limit = %d, %v; want 1", got, err)
	}
	t.Setenv(key, "3")
	if got, err := parallelRequestLimit(); got != 3 || err != nil {
		t.Fatalf("configured limit = %d, %v; want 3", got, err)
	}
}

func TestServiceRejectsInvalidParallelRequestLimit(t *testing.T) {
	const key = "MODELS_MAX_PARALLEL_REQUESTS"
	for _, value := range []string{"", "0", "-1", "1.5", "many", "9999999999999999999999999"} {
		t.Run("value="+value, func(t *testing.T) {
			t.Setenv(key, value)
			_, err := service(manager.Config{Models: []manager.Model{}})
			if err == nil || !strings.Contains(err.Error(), key) {
				t.Fatalf("service error = %v; want an error naming %s", err, key)
			}
		})
	}
}

func TestServiceRoutes(t *testing.T) {
	t.Setenv("MODELS_HISTORY_PATH", filepath.Join(t.TempDir(), "calls.jsonl"))
	t.Setenv("MODELS_MAX_PARALLEL_REQUESTS", "1")
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("unexpected metadata request %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"data":[]}`)
	}))
	defer up.Close()
	cfg := manager.Config{Models: []manager.Model{{Name: "test", Model: "model", Role: "chat", Provider: "openai-compatible", BaseURL: up.URL, Revision: "r1"}}}
	h, err := service(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		host, path string
		code       int
		body       string
	}{
		{"localhost", "/", 200, "Model inventory"},
		{"localhost", "/status", 200, "checked_at"},
		{"models-configurator", "/v1/models", 200, cfg.Models[0].ID()},
		{"unknown.example", "/v1/models", 403, ""},
	} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", "http://"+tc.host+tc.path, nil))
		if w.Code != tc.code || !strings.Contains(w.Body.String(), tc.body) {
			t.Errorf("%+v: %d %s", tc, w.Code, w.Body)
		}
	}
}
