package status

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/history"
)

func TestHistoryPagination(t *testing.T) {
	calls, err := history.Open(filepath.Join(t.TempDir(), "calls.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	RegisterHistory(mux, calls)
	read := func(path string) string {
		t.Helper()
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		if w.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", path, w.Code, w.Body)
		}
		return w.Body.String()
	}

	if got := strings.TrimSpace(read("/history.json")); got != `{"calls":[]}` {
		t.Fatalf("empty history = %q", got)
	}
	if strings.Contains(read("/history"), `id="duration-chart"`) {
		t.Fatal("empty history should not render a chart")
	}
	for i := 1; i <= 52; i++ {
		if err := calls.Record(history.Call{Profile: fmt.Sprint("model-", i)}); err != nil {
			t.Fatal(err)
		}
	}

	top := read("/history?lang=en")
	if !strings.Contains(top, "Load more") || !strings.Contains(top, `name="before" value="3"`) {
		t.Fatalf("first page has no working Load more form")
	}
	old := read("/history?lang=en&before=3")
	if strings.Contains(old, "Load more") || !strings.Contains(old, "model-2") || !strings.Contains(old, "model-1") {
		t.Fatalf("older page has wrong rows or pagination control")
	}
	var page history.Page
	if err := json.Unmarshal([]byte(read("/history.json")), &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Calls) != 50 || page.Calls[0].ID != 52 || page.Calls[49].ID != 3 || page.NextBefore != 3 {
		t.Fatalf("wrong first page, count %d and cursor %d", len(page.Calls), page.NextBefore)
	}
}

func TestHistoryChartMarkup(t *testing.T) {
	calls, err := history.Open(filepath.Join(t.TempDir(), "calls.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	for _, call := range []history.Call{
		{ModelID: "embed.profile.rev", Profile: "embedding", Operation: "embeddings", DurationMS: 25},
		{ModelID: "bge.profile.rev", Profile: "bge", Operation: "bge_encode", DurationMS: 403},
		{ModelID: "chat.profile.rev", Profile: "chat", Operation: "chat", DurationMS: 171},
		{ModelID: "decision.profile.rev", Profile: "decision", Operation: "decision", DurationMS: 16},
	} {
		if err := calls.Record(call); err != nil {
			t.Fatal(err)
		}
	}
	mux := http.NewServeMux()
	RegisterHistory(mux, calls)
	page := httptest.NewRecorder()
	mux.ServeHTTP(page, httptest.NewRequest(http.MethodGet, "/history?lang=ru", nil))
	html := page.Body.String()
	mount, table := strings.Index(html, `id="duration-chart"`), strings.Index(html, "<table>")
	if mount < 0 || table < 0 || mount >= table {
		t.Fatal("duration chart mount should be above the history table")
	}
	for _, fragment := range []string{
		`data-model-id="bge.profile.rev"`,
		`data-model-name="bge"`,
		`data-chart-usage="Тип использования"`,
		`data-operation="bge_encode"`,
		`data-duration-ms="403"`,
		`data-endpoint="POST /v1/chat/completions"`,
		`data-endpoint="POST /v1/embeddings"`,
		`data-endpoint="POST /v1/decisions"`,
		`data-endpoint="POST /v1/bge-m3/encode"`,
		`src="/history-chart.js"`,
		"Длительность вызовов",
	} {
		if !strings.Contains(html, fragment) {
			t.Errorf("history page lacks %q", fragment)
		}
	}
	script := httptest.NewRecorder()
	mux.ServeHTTP(script, httptest.NewRequest(http.MethodGet, "/history-chart.js", nil))
	if script.Code != http.StatusOK || !strings.Contains(script.Header().Get("Content-Type"), "javascript") ||
		script.Body.Len() == 0 || !strings.Contains(script.Body.String(), "licenses.txt") {
		t.Fatal("local chart bundle or its license reference is missing")
	}
	licenses := httptest.NewRecorder()
	mux.ServeHTTP(licenses, httptest.NewRequest(http.MethodGet, "/history-chart-licenses.txt", nil))
	if licenses.Code != http.StatusOK || !strings.Contains(licenses.Body.String(), "recharts@") {
		t.Fatal("local third-party license notices are missing")
	}
}
