package status

import (
	_ "embed"
	"html/template"
	"io"
	"net/http"
)

//go:embed theme.html
var themeHTML string

//go:embed theme.css
var themeCSS string

//go:embed theme.js
var themeJS string

func newPanelTemplate(name, source string, funcs template.FuncMap) *template.Template {
	base := template.Must(template.New(name).Funcs(funcs).Parse(themeHTML))
	return template.Must(base.Parse(source))
}

func RegisterTheme(mux *http.ServeMux) {
	mux.HandleFunc("GET /theme.css", themeAsset("text/css; charset=utf-8", themeCSS))
	mux.HandleFunc("GET /theme.js", themeAsset("text/javascript; charset=utf-8", themeJS))
}

func themeAsset(contentType, content string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		io.WriteString(w, content)
	}
}
