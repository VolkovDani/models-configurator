package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLocalAccess(t *testing.T) {
	for _, tc := range []struct {
		host, origin, site string
		allowed            bool
	}{
		{"127.0.0.1:8317", "", "", true},
		{"[::1]:8317", "", "", true},
		{"models-configurator:8317", "", "", true},
		{"localhost:8317", "http://localhost:8317", "same-origin", true},
		{"untrusted.example:8317", "", "", false},
		{"localhost:8317", "http://untrusted.example", "", false},
		{"localhost:8317", "http://localhost:1234", "", false},
		{"localhost:8317", "null", "", false},
		{"localhost:8317", "", "cross-site", false},
		{"localhost:8317", "", "same-site", false},
	} {
		r := httptest.NewRequest("GET", "http://"+tc.host+"/", nil)
		r.Header.Set("Origin", tc.origin)
		r.Header.Set("Sec-Fetch-Site", tc.site)
		w := httptest.NewRecorder()
		localAccess(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		})).ServeHTTP(w, r)
		if got := w.Code; tc.allowed && got != 204 || !tc.allowed && got != 403 {
			t.Errorf("%+v: status %d", tc, got)
		}
	}
}
