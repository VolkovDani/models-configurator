package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

func TestServiceRoutes(t *testing.T) {
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
