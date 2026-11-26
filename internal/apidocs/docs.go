// Package apidocs serves the OpenAPI specification and bundled Swagger UI.
package apidocs

import (
	"embed"
	"io/fs"
	"net/http"
)

// swagger contains the bundled assets and notices from swagger-ui-dist 5.36.0.
//
//go:embed swagger
var files embed.FS

//go:embed openapi.json
var specification []byte

func Register(mux *http.ServeMux) {
	assets, err := fs.Sub(files, "swagger")
	if err != nil {
		panic(err)
	}
	ui := http.StripPrefix("/docs", http.FileServerFS(assets))
	mux.HandleFunc("GET /docs/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
		ui.ServeHTTP(w, r)
	})
	mux.HandleFunc("GET /openapi.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(specification)
	})
}
