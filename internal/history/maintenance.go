package history

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Lock reserves the history path on platforms that support file locking.
// Keep the sidecar in place so the lock survives replacement of the JSONL file.
func Lock(path string) (release func(), err error) {
	if !lockingSupported {
		return func() {}, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, err
	}
	file, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := tryLock(file); err != nil {
		file.Close()
		return nil, fmt.Errorf("lock history %q: %w", path, err)
	}
	return func() { file.Close() }, nil
}

// Clear moves a nonempty history file into a private backup directory.
// It returns the backup path, or an empty string when there was nothing to move.
func Clear(path string) (string, error) {
	if !lockingSupported {
		return "", errors.New("history clear is unavailable on this platform")
	}
	release, err := Lock(path)
	if err != nil {
		return "", err
	}
	defer release()

	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("history path %q must be a regular file", path)
	}
	if info.Size() == 0 {
		return "", nil
	}

	base := filepath.Base(path)
	dir, err := os.MkdirTemp(filepath.Dir(path), base+".backup-")
	if err != nil {
		return "", err
	}
	backup := filepath.Join(dir, base)
	if err := os.Rename(path, backup); err != nil {
		return "", errors.Join(err, os.Remove(dir))
	}
	return backup, nil
}
