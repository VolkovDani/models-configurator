package status

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/VolkovDani/models-configurator/internal/manager"
)

const (
	npuPromptLimit     = 32768
	npuCompletionLimit = 4096
	npuManifestLimit   = 128 << 10
)

func (s settingsEditor) deriveNPU(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, npuManifestLimit)
	if err := r.ParseMultipartForm(npuManifestLimit); err != nil {
		s.render(w, r, http.StatusBadRequest, "settings_invalid_manifest")
		return
	}
	defer r.MultipartForm.RemoveAll()
	sourceID := r.MultipartForm.Value["profile"]
	var pinned string
	for _, model := range s.cfg.Models {
		if model.Provider == manager.NPUProvider && len(sourceID) == 1 && sourceID[0] == model.ID() {
			pinned = strings.TrimPrefix(model.Digest, "sha256:")
			break
		}
	}
	prompt, promptErr := strconv.Atoi(r.FormValue("prompt"))
	completion, completionErr := strconv.Atoi(r.FormValue("completion"))
	file, _, err := r.FormFile("manifest")
	if err != nil {
		s.render(w, r, http.StatusBadRequest, "settings_invalid_manifest")
		return
	}
	defer file.Close()
	if pinned == "" || promptErr != nil || completionErr != nil || prompt < 1 || prompt > npuPromptLimit || completion < 1 || completion > npuCompletionLimit {
		s.render(w, r, http.StatusBadRequest, "settings_invalid_manifest")
		return
	}
	manifest, digest, err := deriveNPUManifest(file, pinned, prompt, completion)
	if err != nil {
		s.render(w, r, http.StatusBadRequest, "settings_invalid_manifest")
		return
	}
	downloadJSON(w, "npu-"+digest+".json")
	_, _ = w.Write(append(manifest, '\n'))
}

func deriveNPUManifest(r io.Reader, pinned string, prompt, completion int) ([]byte, string, error) {
	var manifest map[string]any
	decoder := json.NewDecoder(r)
	decoder.UseNumber()
	if decoder.Decode(&manifest) != nil || decoder.Decode(new(any)) != io.EOF ||
		manifest["version"] != json.Number("2") || manifest["device"] != "NPU" ||
		manifest["output_format"] != "json-schema-v1" {
		return nil, "", errors.New("expected an NPU version 2 manifest")
	}
	_, digest, err := canonicalManifest(manifest)
	if err != nil || digest != pinned {
		return nil, "", errors.New("manifest does not match pinned profile")
	}
	manifest["max_prompt_tokens"] = prompt
	manifest["max_new_tokens"] = completion
	return canonicalManifest(manifest)
}

// Match the runtime's JSON representation for a pinned v2 manifest.
func canonicalManifest(manifest map[string]any) ([]byte, string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(manifest); err != nil {
		return nil, "", err
	}
	raw := bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
	// encoding/json always escapes these two valid Unicode characters. Skip
	// escaped backslashes so a literal "\\u2028" in a value stays literal.
	canonical := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		if raw[i] == '\\' && i+1 < len(raw) {
			switch {
			case bytes.HasPrefix(raw[i:], []byte(`\u2028`)):
				canonical = append(canonical, "\u2028"...)
				i += 5
				continue
			case bytes.HasPrefix(raw[i:], []byte(`\u2029`)):
				canonical = append(canonical, "\u2029"...)
				i += 5
				continue
			}
			canonical = append(canonical, raw[i])
			i++
		}
		canonical = append(canonical, raw[i])
	}
	digest := sha256.Sum256(canonical)
	return canonical, hex.EncodeToString(digest[:]), nil
}
