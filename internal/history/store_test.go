package history

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestStoreRetentionAndCursor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	store, err := open(path, 4)
	if err != nil {
		t.Fatal(err)
	}
	for range 10 {
		if err := store.Record(Call{Profile: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	first := store.List(0, 2)
	if first.NextBefore != 9 || !reflect.DeepEqual(callIDs(first), []uint64{10, 9}) {
		t.Fatalf("first page = %+v", first)
	}

	// New calls must not shift a cursor that was already issued.
	store, err = open(path, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Record(Call{Profile: "test"}); err != nil {
		t.Fatal(err)
	}
	second := store.List(first.NextBefore, 2)
	if second.NextBefore != 0 || !reflect.DeepEqual(callIDs(second), []uint64{8}) {
		t.Fatalf("second page = %+v", second)
	}
	if got := callIDs(store.List(0, 10)); !reflect.DeepEqual(got, []uint64{11, 10, 9, 8}) {
		t.Fatalf("retained calls = %v", got)
	}
}

func TestStoreRepairsUnterminatedTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":1}\n{\"id\":"), 0600); err != nil {
		t.Fatal(err)
	}
	store, err := open(path, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Record(Call{Profile: "test"}); err != nil {
		t.Fatal(err)
	}
	store, err = open(path, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got := callIDs(store.List(0, 10)); !reflect.DeepEqual(got, []uint64{2, 1}) {
		t.Fatalf("recovered calls = %v", got)
	}
}

func TestStoreSerializesConcurrentCalls(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	store, err := open(path, 64)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 24 {
		wg.Go(func() {
			if err := store.Record(Call{Profile: "test"}); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	store, err = open(path, 64)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]uint64, 0, 24)
	for id := uint64(24); id > 0; id-- {
		want = append(want, id)
	}
	if got := callIDs(store.List(0, 64)); !reflect.DeepEqual(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
}

func callIDs(page Page) []uint64 {
	ids := make([]uint64, 0, len(page.Calls))
	for _, call := range page.Calls {
		ids = append(ids, call.ID)
	}
	return ids
}
