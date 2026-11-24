package manager

import (
	"context"
	"errors"
	"slices"
	"sync"
)

const maxWaiting = 128

var errQueueFull = errors.New("inference queue is full")

type waiter struct {
	socket string
	ready  chan struct{}
}

type dispatcher struct {
	mu          sync.Mutex
	maxParallel int
	active      int
	busySockets map[string]bool
	waiting     []*waiter // arrival order
}

func newDispatcher(maxParallel int) *dispatcher {
	return &dispatcher{maxParallel: maxParallel, busySockets: make(map[string]bool)}
}

// A free slot can be used immediately: all older waiters either have a
// reserved slot or are waiting for a busy socket.
func (d *dispatcher) acquire(ctx context.Context, socket string) error {
	d.mu.Lock()
	if err := ctx.Err(); err != nil {
		d.mu.Unlock()
		return err
	}
	if d.active < d.maxParallel && !d.busySockets[socket] {
		d.reserve(socket)
		d.mu.Unlock()
		return nil
	}
	if len(d.waiting) == maxWaiting {
		d.mu.Unlock()
		return errQueueFull
	}
	w := &waiter{socket: socket, ready: make(chan struct{})}
	d.waiting = append(d.waiting, w)
	d.mu.Unlock()

	select {
	case <-w.ready:
	case <-ctx.Done():
	}
	if err := ctx.Err(); err != nil {
		d.mu.Lock()
		if i := slices.Index(d.waiting, w); i >= 0 {
			d.waiting = slices.Delete(d.waiting, i, i+1)
		} else {
			// Release raced with cancellation and already reserved our slot.
			d.releaseLocked(socket)
		}
		d.mu.Unlock()
		return err
	}
	return nil
}

func (d *dispatcher) reserve(socket string) {
	d.active++
	if socket != "" {
		d.busySockets[socket] = true
	}
}

func (d *dispatcher) release(socket string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.releaseLocked(socket)
}

func (d *dispatcher) releaseLocked(socket string) {
	d.active--
	delete(d.busySockets, socket)
	for i, w := range d.waiting {
		if !d.busySockets[w.socket] {
			d.waiting = slices.Delete(d.waiting, i, i+1)
			d.reserve(w.socket)
			close(w.ready)
			break
		}
	}
}
