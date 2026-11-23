package main

import (
	"net"
	"net/http"
	"strings"
)

// Host checks also apply to API routes because they can trigger provider calls.
func localAccess(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		if name, _, err := net.SplitHostPort(host); err == nil {
			host = name
		}
		ip := net.ParseIP(strings.Trim(host, "[]"))
		trusted := strings.EqualFold(host, "localhost") || strings.EqualFold(host, "models-configurator") || ip.IsLoopback()
		origins := r.Header.Values("Origin")
		origin := r.Header.Get("Origin")
		site := r.Header.Get("Sec-Fetch-Site")
		if !trusted || len(origins) > 1 || (origin != "" && !strings.EqualFold(origin, "http://"+r.Host)) ||
			(site != "" && site != "same-origin" && site != "none") {
			http.Error(w, http.StatusText(http.StatusForbidden), http.StatusForbidden)
			return
		}
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}
