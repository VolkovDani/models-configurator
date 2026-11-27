//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/VolkovDani/models-configurator/internal/history"
)

func TestHistoryClearCLI(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "calls.jsonl")
	t.Setenv("MODELS_HISTORY_PATH", path)
	run := func(args ...string) (string, error) {
		command := exec.CommandContext(t.Context(), os.Args[0], append([]string{"-test.run=^TestHistoryCLIProcess$", "--"}, args...)...)
		command.Dir = dir
		output, err := command.CombinedOutput()
		return string(output), err
	}

	original := []byte("{\"id\":1}\n")
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	if output, err := run("history", "clear", "unexpected"); err == nil || !strings.Contains(output, "history clear") {
		t.Fatalf("unexpected argument: %v\n%s", err, output)
	}
	if output, err := run("-help"); err != nil || !strings.Contains(output, "history clear") {
		t.Fatalf("top-level help: %v\n%s", err, output)
	}
	release, err := history.Lock(path)
	if err != nil {
		t.Fatal(err)
	}
	lockedOutput, lockedErr := run("history", "clear")
	release()
	if lockedErr == nil || !strings.Contains(lockedOutput, "stop the service") {
		t.Fatalf("clear locked history: %v\n%s", lockedErr, lockedOutput)
	}
	output, err := run("history", "clear")
	if err != nil {
		t.Fatalf("clear history: %v\n%s", err, output)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("active history should be absent after clearing, got %v", err)
	}
	backups, err := filepath.Glob(path + ".backup-*/calls.jsonl")
	if err != nil || len(backups) != 1 || !strings.Contains(output, backups[0]) {
		t.Fatalf("expected the command to report one backup, got %v, %v\n%s", backups, err, output)
	}
	if saved, err := os.ReadFile(backups[0]); err != nil || !bytes.Equal(saved, original) {
		t.Fatalf("backup contains %q, error %v", saved, err)
	}
	if output, err := run("history", "clear"); err != nil || !strings.Contains(output, "already empty") {
		t.Fatalf("clear empty history: %v\n%s", err, output)
	}
	if after, err := filepath.Glob(path + ".backup-*/calls.jsonl"); err != nil || len(after) != 1 || after[0] != backups[0] {
		t.Fatalf("repeating clear should preserve the existing backup, got %v, %v", after, err)
	}
}

func TestHistoryCLIProcess(t *testing.T) {
	if len(os.Args) < 4 || os.Args[2] != "--" {
		return
	}
	os.Args = append(os.Args[:1], os.Args[3:]...)
	main()
	os.Exit(0)
}
