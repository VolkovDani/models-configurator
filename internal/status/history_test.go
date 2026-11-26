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
