//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package history

import (
	"errors"
	"os"
	"syscall"
)

const lockingSupported = true

func tryLock(file *os.File) error {
	err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return errors.New("history is in use, stop the service and retry")
	}
	return err
}
