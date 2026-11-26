package manager

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/VolkovDani/models-configurator/internal/history"
)

func bgeModel(base string) Model {
	return Model{Name: "intel", Provider: "openvino-bge", Model: "BAAI/bge-m3", Role: "embeddings", Digest: "sha256:" + strings.Repeat("b", 64), BaseURL: base}
}

func TestBGEConfigurationAndAdmission(t *testing.T) {
	m := bgeModel(socketRuntime(t, func(http.ResponseWriter, *http.Request) { t.Error("unexpected worker call") }))
	other := Model{Name: "direct", Provider: "openai-compatible", BaseURL: "http://localhost:4321/v1", Role: "embeddings", Model: "other", Revision: "r1"}
	h, err := New(Config{Models: []Model{m, other}})
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Model){func(v *Model) { v.BaseURL = "http://localhost" }, func(v *Model) { v.Model = "other" }, func(v *Model) { v.Role = "chat" }, func(v *Model) { v.Digest = "latest" }, func(v *Model) { v.KeyEnv = "KEY" }, func(v *Model) { v.Revision = "r1" }} {
		bad := m
		change(&bad)
		if bad.validate() == nil {
			t.Errorf("accepted invalid BGE config: %+v", bad)
		}
	}
	for _, tc := range []struct {
		path, id, fields string
		code             int
	}{
		{"/v1/embeddings", m.ID(), `"input":"hello","dimensions":2`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":["x"],"modes":["dense","dense"]`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":["x"],"modes":["other"]`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":["x"],"modes":[]`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":"x","modes":["dense"]`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":["x","x","x","x","x"],"modes":["sparse"]`, 400},
		{"/v1/bge-m3/encode", m.ID(), `"input":[" "],"modes":["sparse"]`, 400},
		{"/v1/bge-m3/encode", other.ID(), `"input":["x"],"modes":["sparse"]`, 404},
	} {
		w := perform(h, "POST", tc.path, fmt.Sprintf(`{"model":%q,%s}`, tc.id, tc.fields))
		if w.Code != tc.code {
			t.Errorf("%s %s: %d", tc.path, tc.fields, w.Code)
		}
	}
}

func TestBGEWorkerContract(t *testing.T) {
	pin := strings.Repeat("b", 64)
	vec := make([]float32, 1024)
	vec[0] = 1
	for _, tc := range []struct {
		scenario      string
		status, calls int
	}{
		{"dense", 200, 2}, {"rich", 200, 2}, {"sparse-only", 200, 2}, {"empty-sparse", 200, 2}, {"status-pin", 409, 1}, {"status-device", 409, 1}, {"not-loaded", 503, 1}, {"result-pin", 409, 2}, {"cpu-result", 409, 2}, {"changed-gpu", 409, 2}, {"input-limit", 413, 2}, {"worker-conflict", 409, 2}, {"worker-private", 502, 2}, {"bad-colbert", 502, 2}, {"negative-weight", 502, 2}, {"bad-token", 502, 2}, {"too-many-tokens", 502, 2}, {"zero-dense", 502, 2}, {"unknown-field", 502, 2}, {"missing-field", 502, 2}, {"missing-row", 502, 2},
	} {
		t.Run(tc.scenario, func(t *testing.T) {
			calls := 0
			modes := []string{"dense", "sparse", "colbert"}
			switch tc.scenario {
			case "dense":
				modes = []string{"dense"}
			case "sparse-only", "empty-sparse":
				modes = []string{"sparse"}
			}
			m := bgeModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
				calls++
				profile, device := pin, "GPU.0"
				switch r.Method + " " + r.URL.Path {
				case "GET /status":
					if tc.scenario == "status-pin" {
						profile = strings.Repeat("c", 64)
					}
					if tc.scenario == "status-device" {
						device = "AUTO:GPU,CPU"
					}
					json.NewEncoder(w).Encode(map[string]any{"profile": profile, "device": device, "loaded": tc.scenario != "not-loaded"})
				case "POST /v1/encode":
					var got struct {
						Profile      string
						Input, Modes []string
					}
					dec := json.NewDecoder(r.Body)
					dec.DisallowUnknownFields()
					if dec.Decode(&got) != nil || got.Profile != pin || !slices.Equal(got.Input, []string{"private input"}) || !slices.Equal(got.Modes, modes) {
						t.Errorf("wrong encode request %+v", got)
					}
					row := map[string]any{}
					for _, mode := range modes {
						switch mode {
						case "dense":
							row[mode] = vec
						case "colbert":
							row[mode] = [][]float32{vec}
						case "sparse":
							row[mode] = map[string]float32{"250001": 0.5}
						}
					}
					switch tc.scenario {
					case "result-pin":
						profile = strings.Repeat("c", 64)
					case "cpu-result":
						device = "CPU"
					case "changed-gpu":
						device = "GPU.1"
					case "input-limit":
						http.Error(w, "private input", 413)
						return
					case "worker-conflict":
						http.Error(w, "private input", 409)
						return
					case "worker-private":
						http.Error(w, "private input", 500)
						return
					case "bad-colbert":
						row["colbert"] = [][]float32{{1}}
					case "negative-weight":
						row["sparse"] = map[string]float32{"9": -1}
					case "bad-token":
						row["sparse"] = map[string]float32{"250002": 1}
					case "too-many-tokens":
						rows := make([][]float32, 129)
						for i := range rows {
							rows[i] = vec
						}
						row["colbert"] = rows
					case "zero-dense":
						row["dense"] = make([]float32, 1024)
					case "unknown-field":
						row["other"] = "private input"
					case "missing-field":
						delete(row, "dense")
					case "empty-sparse":
						row["sparse"] = map[string]float32{}
					}
					rows := []map[string]any{row}
					if tc.scenario == "missing-row" {
						rows = nil
					}
					json.NewEncoder(w).Encode(map[string]any{"profile": profile, "device": device, "data": rows, "input_tokens": 11})
				default:
					t.Errorf("unexpected worker request %s", r.URL.Path)
				}
			}))
			var recorded []history.Call
			h, err := NewWithHistory(Config{Models: []Model{m}}, 1, func(call history.Call) { recorded = append(recorded, call) })
			if err != nil {
				t.Fatal(err)
			}
			path := "/v1/bge-m3/encode"
			operation := "bge_encode"
			body := map[string]any{"model": m.ID(), "input": []string{"private input"}, "modes": modes}
			if tc.scenario == "dense" {
				path = "/v1/embeddings"
				operation = "embeddings"
				delete(body, "modes")
				body["dimensions"] = 1024
			}
			raw, _ := json.Marshal(body)
			w := perform(h, "POST", path, string(raw))
			if len(recorded) != 1 || recorded[0].Operation != operation || recorded[0].Status != tc.status {
				t.Fatalf("history = %+v, want one %s call at %d", recorded, operation, tc.status)
			}
			if w.Code != tc.status || calls != tc.calls || strings.Contains(w.Body.String(), "private input") {
				t.Fatalf("status %d calls %d", w.Code, calls)
			}
			if tc.status == 200 {
				assertHistoryUsage(t, recorded[0], `{"input_tokens":11}`)
				var result struct {
					Model string
					Data  []map[string]json.RawMessage
				}
				if json.Unmarshal(w.Body.Bytes(), &result) != nil || result.Model != m.ID() || len(result.Data) != 1 {
					t.Fatal("bad response")
				}
				row := result.Data[0]
				if tc.scenario == "dense" {
					if row["embedding"] == nil || string(row["index"]) != "0" {
						t.Error("bad embedding shape")
					}
				} else {
					if len(row) != len(modes) {
						t.Fatal("returned unexpected modes", row)
					}
					for _, mode := range modes {
						if row[mode] == nil {
							t.Errorf("missing %s", mode)
						}
					}
				}
			}
		})
	}
}

func TestBGEDenseBatchAndRichShareSocket(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var probes atomic.Int32
	pin := strings.Repeat("b", 64)
	input := make([]string, 32)
	rows := make([]map[string]any, 32)
	for i := range input {
		input[i] = fmt.Sprintf("source-%d", i)
		v := make(vector, 1024)
		v[i] = 1
		rows[i] = map[string]any{"dense": v}
	}
	m := bgeModel(socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/status" {
			probes.Add(1)
			fmt.Fprintf(w, `{"profile":%q,"device":"GPU","loaded":true}`, pin)
			return
		}
		var body struct{ Input []string }
		if json.NewDecoder(r.Body).Decode(&body) != nil || !slices.Equal(body.Input, input) {
			http.Error(w, "unexpected input", 400)
			return
		}
		close(started)
		<-release
		json.NewEncoder(w).Encode(map[string]any{"profile": pin, "device": "GPU", "data": rows})
	}))
	h, err := NewWithConcurrency(Config{Models: []Model{m}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(map[string]any{"model": m.ID(), "input": input})
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- perform(h, "POST", "/v1/embeddings", string(raw)) }()
	defer func() {
		close(release)
		res := <-done
		var got struct{ Data []embeddingRow }
		if res.Code != 200 || json.Unmarshal(res.Body.Bytes(), &got) != nil || len(got.Data) != 32 {
			t.Errorf("32 input batch: status %d", res.Code)
			return
		}
		for i, row := range got.Data {
			if row.Index == nil || *row.Index != i || len(row.Embedding) != 1024 || row.Embedding[i] != 1 {
				t.Errorf("invalid batch row %d", i)
			}
		}
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("batch did not reach worker")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/bge-m3/encode", strings.NewReader(fmt.Sprintf(`{"model":%q,"input":["x"],"modes":["sparse"]}`, m.ID())))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 504 || probes.Load() != 1 {
		t.Errorf("rich request on busy socket: status %d, probes %d", w.Code, probes.Load())
	}
}
