package manager

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/VolkovDani/models-configurator/internal/history"
)

func TestDefaultQueueSharesCapacityBetweenChatAndEmbeddings(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/chat/completions" {
			close(started)
			<-release
			fmt.Fprint(w, `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer server.Close()
	defer close(release)
	chat := Model{Name: "chat", Provider: "openai-compatible", Model: "s", Revision: "1", Role: "chat", BaseURL: server.URL}
	embed := chat
	embed.Name, embed.Role = "embed", "embeddings"
	h, err := New(Config{Models: []Model{chat, embed}})
	if err != nil {
		t.Fatal(err)
	}
	go perform(h, "POST", "/v1/chat/completions", chatRequest(chat.ID()))
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("chat did not start")
	}
	w := httptest.NewRecorder()
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	req := httptest.NewRequestWithContext(ctx, "POST", "/v1/embeddings", strings.NewReader(fmt.Sprintf(`{"model":%q,"input":"a"}`, embed.ID())))
	h.ServeHTTP(w, req)
	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("embedding while chat slot is occupied: got %d, want 504", w.Code)
	}
	if w := perform(h, "GET", "/v1/models", ""); w.Code != http.StatusOK {
		t.Fatalf("inventory while slot is occupied: %d", w.Code)
	}
}

func TestDispatcherAdmitsOldestEligibleAndRemovesCanceledWaiters(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := newDispatcher(2)
		ctx := t.Context()
		if err := q.acquire(ctx, "npu"); err != nil {
			t.Fatal(err)
		}
		wait := func(ctx context.Context, socket string) chan error {
			result := make(chan error, 1)
			go func() { result <- q.acquire(ctx, socket) }()
			synctest.Wait()
			return result
		}
		canceled, cancel := context.WithCancel(ctx)
		first := wait(canceled, "npu")
		next := wait(ctx, "npu")
		last := wait(ctx, "npu")
		independent := wait(ctx, "other-npu")
		if err := <-independent; err != nil {
			t.Fatal(err)
		}
		other := wait(ctx, "")
		assertWaiting := func(result chan error) {
			t.Helper()
			select {
			case err := <-result:
				t.Fatalf("unexpected admission: %v", err)
			default:
			}
		}
		assertWaiting(first)
		assertWaiting(next)
		assertWaiting(last)
		assertWaiting(other)
		cancel()
		if err := <-first; !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
		q.release("other-npu")
		if err := <-other; err != nil {
			t.Fatal(err)
		}
		q.release("npu")
		if err := <-next; err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		assertWaiting(last)
		q.release("npu")
		if err := <-last; err != nil {
			t.Fatal(err)
		}
		q.release("npu")
		q.release("")
	})
}

func TestInferenceQueueOverflowTimeoutAndRelease(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := newDispatcher(1)
		var recorded []history.Call
		q.record = func(call history.Call) { recorded = append(recorded, call) }
		if err := q.acquire(t.Context(), ""); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		p := provider{model: Model{Role: "chat"}}
		invoke := func(context.Context) (any, error) {
			t.Error("queued request reached provider")
			return nil, nil
		}
		call := func(ctx context.Context) int {
			req := httptest.NewRequestWithContext(ctx, "POST", "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			q.run(w, req, p, "chat", invoke)
			return w.Code
		}
		done := make(chan int, 128)
		for range 128 {
			go func() { done <- call(ctx) }()
		}
		synctest.Wait()
		if got := call(t.Context()); got != http.StatusTooManyRequests {
			t.Fatalf("overflow: got %d", got)
		}
		cancel()
		for range 128 {
			if got := <-done; got != http.StatusRequestTimeout {
				t.Fatalf("canceled: got %d", got)
			}
		}
		start := time.Now()
		if got := call(t.Context()); got != http.StatusGatewayTimeout || time.Since(start) != 2*time.Minute {
			t.Fatalf("queue timeout: got %d after %v", got, time.Since(start))
		}
		q.release("")
		req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		w := httptest.NewRecorder()
		q.run(w, req, p, "chat", func(context.Context) (any, error) { return nil, errors.New("provider failed") })
		if w.Code != http.StatusBadGateway {
			t.Fatal(w.Code)
		}
		if len(recorded) != 1 || recorded[0].Status != 502 || recorded[0].Outcome != "failed" {
			t.Fatalf("history = %+v, want one failed dispatch", recorded)
		}
		if err := q.acquire(t.Context(), ""); err != nil {
			t.Fatal(err)
		}
		q.release("")
	})
}

func TestInferenceHistoryRecordsActiveTimeoutAndCancellation(t *testing.T) {
	for _, tc := range []struct {
		name string
		stop func(context.CancelFunc)
	}{
		{"timed_out", func(context.CancelFunc) { time.Sleep(2 * time.Minute) }},
		{"canceled", func(cancel context.CancelFunc) { cancel() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				q := newDispatcher(1)
				done := make(chan history.Call, 1)
				q.record = func(call history.Call) { done <- call }
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				req := httptest.NewRequestWithContext(ctx, "POST", "/v1/chat/completions", nil)
				go q.run(httptest.NewRecorder(), req, provider{}, "chat", func(ctx context.Context) (any, error) {
					<-ctx.Done()
					return nil, ctx.Err()
				})
				synctest.Wait()
				tc.stop(cancel)
				if call := <-done; call.Outcome != tc.name || call.Status != http.StatusBadGateway {
					t.Fatalf("history = %+v, want %s", call, tc.name)
				}
			})
		})
	}
}

func TestNPUQueueAliasesAllowConcurrentEmbeddings(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var healthCalls atomic.Int32
	const profile = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	base := socketRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			healthCalls.Add(1)
			fmt.Fprintf(w, `{"profile":%q,"device":"NPU"}`, profile)
			return
		}
		if healthCalls.Load() == 1 {
			close(started)
			<-release
		}
		fmt.Fprintf(w, `{"profile":%q,"device":"NPU","content":"{}"}`, profile)
	})
	embedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer embedServer.Close()
	m := npuModel(base)
	alias := m
	alias.Name = "alias"
	socket, err := socketPath(base)
	if err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(filepath.Dir(socket), "alias.sock")
	alias.BaseURL = "unix://" + strings.Replace(link, "alias.sock", "%61lias.sock", 1)
	embed := Model{Name: "embed", Provider: "openai-compatible", Model: "s", Revision: "1", Role: "embeddings", BaseURL: embedServer.URL}
	h, err := NewWithConcurrency(Config{Models: []Model{m, alias, embed}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	// A deployment may start the manager before mounting or linking the socket.
	if err := os.Symlink(socket, link); err != nil {
		t.Fatal(err)
	}
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- perform(h, "POST", "/v1/chat/completions", chatRequest(m.ID())) }()
	defer func() {
		close(release)
		if w := <-done; w.Code != 200 {
			t.Errorf("first request: %d %s", w.Code, w.Body)
		}
	}()
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("NPU request did not start")
	}

	invalidCtx, invalidCancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer invalidCancel()
	invalid := strings.Replace(chatRequest(alias.ID()), `"stream":false`, `"temperature":0`, 1)
	wInvalid := httptest.NewRecorder()
	h.ServeHTTP(wInvalid, httptest.NewRequestWithContext(invalidCtx, "POST", "/v1/chat/completions", strings.NewReader(invalid)))
	if wInvalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid NPU options while busy: %d %s", wInvalid.Code, wInvalid.Body)
	}

	// Keep the NPU request active throughout both calls. An alias must time
	// out without even probing /health; another provider has a spare slot.
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequestWithContext(ctx, "POST", "/v1/chat/completions", strings.NewReader(chatRequest(alias.ID()))))
	if w.Code != 504 || healthCalls.Load() != 1 || !strings.Contains(w.Body.String(), `"code":"TIMEOUT"`) {
		t.Fatalf("alias response %d %s, %d health calls", w.Code, w.Body, healthCalls.Load())
	}
	ctx, cancel = context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequestWithContext(ctx, "POST", "/v1/embeddings", strings.NewReader(fmt.Sprintf(`{"model":%q,"input":"a"}`, embed.ID()))))
	if w.Code != 200 {
		t.Fatalf("independent provider: %d %s", w.Code, w.Body)
	}
}

func TestInferenceDeadlineIncludesQueueWait(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := newDispatcher(1)
		if err := q.acquire(t.Context(), ""); err != nil {
			t.Fatal(err)
		}
		p := provider{model: Model{Role: "chat"}}
		done := make(chan time.Duration, 1)
		go func() {
			req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
			q.run(httptest.NewRecorder(), req, p, "chat", func(ctx context.Context) (any, error) {
				deadline, _ := ctx.Deadline()
				done <- time.Until(deadline)
				return nil, nil
			})
		}()
		synctest.Wait()
		time.Sleep(90 * time.Second)
		q.release("")
		if remaining := <-done; remaining != 30*time.Second {
			t.Fatalf("remaining provider budget: %s", remaining)
		}
	})
}

func TestNewWithConcurrencyRejectsNonpositiveLimits(t *testing.T) {
	for _, limit := range []int{-1, 0} {
		if _, err := NewWithConcurrency(Config{Models: []Model{}}, limit); err == nil {
			t.Fatalf("accepted limit %d", limit)
		}
	}
}

func TestDispatcherReturnsSlotGrantedDuringCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := newDispatcher(1)
		if err := q.acquire(t.Context(), "socket"); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		done := make(chan error, 1)
		go func() { done <- q.acquire(ctx, "socket") }()
		synctest.Wait()
		// Hold the mutex so cancellation cannot remove the waiter before
		// the active call releases the socket and reserves it for the waiter.
		q.mu.Lock()
		cancel()
		q.releaseLocked("socket")
		q.mu.Unlock()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("waiter after cancellation: %v", err)
		}
		if err := q.acquire(t.Context(), "socket"); err != nil {
			t.Fatal(err)
		}
		q.release("socket")
	})
}
