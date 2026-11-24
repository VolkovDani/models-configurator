package manager

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const BGEProvider = "openvino-bge"
const bgeDimensions = 1024
const bgeMaxTokens = 128

var gpuDevice = regexp.MustCompile(`^GPU(\.[0-9]+)?$`)
var errBGEOutput = errors.New("invalid BGE worker response")

type bgeRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
	Modes []string `json:"modes"`
}

func (in bgeRequest) valid() bool {
	if len(in.Input) > 4 || !(embeddingRequest{Input: in.Input}).valid() || len(in.Modes) == 0 || len(in.Modes) > 3 {
		return false
	}
	for i, m := range in.Modes {
		if (m != "dense" && m != "sparse" && m != "colbert") || slices.Contains(in.Modes[:i], m) {
			return false
		}
	}
	return true
}

type bgeBinding struct {
	Profile string `json:"profile"`
	Device  string `json:"device"`
}

func (b bgeBinding) check(pin string) error {
	if len(b.Profile) != 64 || !digestPattern.MatchString(b.Profile) || b.Device == "" {
		return errBGEOutput
	}
	if !gpuDevice.MatchString(b.Device) || (pin != "" && b.Profile != pin) {
		return upstreamStatus(http.StatusConflict)
	}
	return nil
}

type bgeStatus struct {
	bgeBinding
	Loaded *bool `json:"loaded"`
}

func (p upstream) bgeStatus(ctx context.Context, pin string) (bgeStatus, error) {
	var s bgeStatus
	if err := p.requestBounded(ctx, http.MethodGet, "/status", nil, &s, 4096); err != nil {
		return s, err
	}
	if s.Loaded == nil {
		return s, errBGEOutput
	}
	return s, s.check(pin)
}

// BGEStatus only fetches passive metadata over the configured local socket.
func BGEStatus(ctx context.Context, base string) (profile string, loaded bool, err error) {
	p, err := socketUpstream(base)
	if err != nil {
		return "", false, err
	}
	s, err := p.bgeStatus(ctx, "")
	if err != nil {
		return "", false, err
	}
	return s.Profile, *s.Loaded, nil
}

type bgeRow struct {
	Dense   vector              `json:"dense,omitempty"`
	Sparse  *map[string]float32 `json:"sparse,omitempty"`
	Colbert []vector            `json:"colbert,omitempty"`
}

func validBGEVector(v vector) bool {
	return len(v) == bgeDimensions && slices.ContainsFunc(v, func(x float32) bool { return x != 0 })
}

func decodeBGERow(fields map[string]json.RawMessage, modes []string) (bgeRow, error) {
	var row bgeRow
	if len(fields) != len(modes) {
		return row, errBGEOutput
	}
	for _, mode := range modes {
		raw := fields[mode]
		switch mode {
		case "dense":
			if json.Unmarshal(raw, &row.Dense) != nil || !validBGEVector(row.Dense) {
				return row, errBGEOutput
			}
		case "sparse":
			if json.Unmarshal(raw, &row.Sparse) != nil || row.Sparse == nil || len(*row.Sparse) > bgeMaxTokens {
				return row, errBGEOutput
			}
			for token, weight := range *row.Sparse {
				id, err := strconv.Atoi(token)
				if err != nil || id < 0 || id > 250001 || strconv.Itoa(id) != token || weight <= 0 {
					return row, errBGEOutput
				}
			}
		case "colbert":
			if json.Unmarshal(raw, &row.Colbert) != nil || len(row.Colbert) == 0 || len(row.Colbert) > bgeMaxTokens {
				return row, errBGEOutput
			}
			for _, v := range row.Colbert {
				if !validBGEVector(v) {
					return row, errBGEOutput
				}
			}
		}
	}
	return row, nil
}

type bgeResult struct {
	Model string   `json:"model"`
	Data  []bgeRow `json:"data"`
}

func (p provider) bgeEncode(ctx context.Context, input, modes []string) (bgeResult, error) {
	var result bgeResult
	pin := strings.TrimPrefix(p.model.Digest, "sha256:")
	s, err := p.bgeStatus(ctx, pin)
	if err != nil {
		return result, err
	}
	if !*s.Loaded {
		return result, upstreamStatus(http.StatusServiceUnavailable)
	}
	body := struct {
		Profile string   `json:"profile"`
		Input   []string `json:"input"`
		Modes   []string `json:"modes"`
	}{pin, input, modes}
	var resp struct {
		bgeBinding
		Data []map[string]json.RawMessage `json:"data"`
	}
	if err := p.requestBounded(ctx, http.MethodPost, "/v1/encode", body, &resp, 16<<20); err != nil {
		return result, err
	}
	if err := resp.check(pin); err != nil {
		return result, err
	}
	if resp.Device != s.Device {
		return result, upstreamStatus(http.StatusConflict)
	}
	if len(resp.Data) != len(input) {
		return result, errBGEOutput
	}
	rows := make([]bgeRow, len(input))
	for i, fields := range resp.Data {
		rows[i], err = decodeBGERow(fields, modes)
		if err != nil {
			return result, err
		}
	}
	return bgeResult{Model: p.model.ID(), Data: rows}, nil
}
