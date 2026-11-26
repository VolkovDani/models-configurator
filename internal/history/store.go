// Package history stores local inference metadata.
package history

import (
	"bufio"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"time"
)

const retention = 10_000

type Call struct {
	ID           uint64    `json:"id"`
	ModelID      string    `json:"model_id"`
	Profile      string    `json:"profile"`
	Provider     string    `json:"provider"`
	Operation    string    `json:"operation"`
	StartedAt    time.Time `json:"started_at"`
	DurationMS   int64     `json:"duration_ms"`
	Status       int       `json:"status"`
	Outcome      string    `json:"outcome"`
	InputTokens  *int      `json:"input_tokens,omitempty"`
	OutputTokens *int      `json:"output_tokens,omitempty"`
}

func (call Call) clone() Call {
	if call.InputTokens != nil {
		call.InputTokens = new(*call.InputTokens)
	}
	if call.OutputTokens != nil {
		call.OutputTokens = new(*call.OutputTokens)
	}
	return call
}

type Page struct {
	Calls      []Call `json:"calls"`
	NextBefore uint64 `json:"next_before,omitempty"`
}

// Store is owned by one service process. It appends synced records until
// compaction can reclaim a batch of expired entries with one atomic rename.
type Store struct {
	mu        sync.Mutex
	path      string
	retention int
	size      int64
	calls     []Call
}

func Open(path string) (*Store, error) {
	return open(path, retention)
}

func open(path string, retention int) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := f.Chmod(0600); err != nil {
		return nil, err
	}

	s := &Store{path: path, retention: retention}
	reader := bufio.NewReader(f)
	var lastID uint64
	for {
		line, err := reader.ReadSlice('\n')
		if err == io.EOF {
			// A process interrupted mid-append can leave an incomplete tail.
			if len(line) != 0 {
				if err := f.Truncate(s.size); err != nil {
					return nil, err
				}
			}
			return s, nil
		}
		if err != nil {
			return nil, err
		}
		var call Call
		if json.Unmarshal(line, &call) != nil || call.ID <= lastID || len(s.calls) == 2*retention {
			return nil, fmt.Errorf("invalid history record at byte %d", s.size)
		}
		lastID = call.ID
		s.calls = append(s.calls, call)
		s.size += int64(len(line))
	}
}

func (s *Store) Record(call Call) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	call = call.clone()
	call.ID = 1
	if n := len(s.calls); n != 0 {
		call.ID = s.calls[n-1].ID + 1
	}
	if len(s.calls) >= 2*s.retention {
		if err := s.compact(); err != nil {
			return err
		}
	}
	line, err := json.Marshal(call)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	f, err := os.OpenFile(s.path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if info.Size() != s.size {
		return errors.New("history file changed while the service was running")
	}
	if _, err := f.Write(line); err != nil {
		return errors.Join(err, f.Truncate(s.size))
	}
	if err := f.Sync(); err != nil {
		return errors.Join(err, f.Truncate(s.size))
	}
	s.size += int64(len(line))
	s.calls = append(s.calls, call)
	return nil
}

func (s *Store) compact() error {
	keep := s.calls[len(s.calls)-s.retention:]
	f, err := os.CreateTemp(filepath.Dir(s.path), ".history-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	size, err := writeCalls(f, keep)
	if closeErr := f.Close(); err != nil || closeErr != nil {
		return errors.Join(err, closeErr)
	}
	if err := os.Rename(f.Name(), s.path); err != nil {
		return err
	}
	copy(s.calls, keep)
	clear(s.calls[len(keep):])
	s.calls = s.calls[:len(keep)]
	s.size = size
	return nil
}

func writeCalls(f *os.File, calls []Call) (int64, error) {
	buf := bufio.NewWriter(f)
	encoder := json.NewEncoder(buf)
	for _, call := range calls {
		if err := encoder.Encode(call); err != nil {
			return 0, err
		}
	}
	if err := buf.Flush(); err != nil {
		return 0, err
	}
	if err := f.Sync(); err != nil {
		return 0, err
	}
	return f.Seek(0, io.SeekCurrent)
}

func (s *Store) List(before uint64, limit int) Page {
	s.mu.Lock()
	defer s.mu.Unlock()
	start := max(0, len(s.calls)-s.retention)
	end := len(s.calls)
	if before != 0 {
		end, _ = slices.BinarySearchFunc(s.calls, before, func(call Call, id uint64) int {
			return cmp.Compare(call.ID, id)
		})
	}
	count := min(limit, max(0, end-start))
	page := Page{Calls: make([]Call, 0, count)}
	for i := end - 1; i >= end-count; i-- {
		page.Calls = append(page.Calls, s.calls[i].clone())
	}
	if count > 0 && end-count > start {
		page.NextBefore = s.calls[end-count].ID
	}
	return page
}
