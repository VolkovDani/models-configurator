//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd

package history

import (
	"errors"
	"os"
)

const lockingSupported = false

func tryLock(*os.File) error {
	return errors.New("history file locking is unavailable on this platform")
}
