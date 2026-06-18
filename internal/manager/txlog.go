package manager

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	txMagic   = "DFTX"
	txVersion = uint16(1)
)

func ParseTransactionFile(path string, stagingRoot string) ([]TransactionOp, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("read tx magic: %w", err)
	}
	if string(magic) != txMagic {
		return nil, fmt.Errorf("bad tx magic %q", string(magic))
	}
	var version uint16
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read tx version: %w", err)
	}
	if version != txVersion {
		return nil, fmt.Errorf("unsupported tx version %d", version)
	}
	var flags uint16
	if err := binary.Read(r, binary.LittleEndian, &flags); err != nil {
		return nil, fmt.Errorf("read tx flags: %w", err)
	}
	if flags != 0 {
		return nil, fmt.Errorf("unsupported tx flags %d", flags)
	}
	var count uint32
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read tx op count: %w", err)
	}

	ops := make([]TransactionOp, 0, count)
	for i := uint32(0); i < count; i++ {
		opByte, err := r.ReadByte()
		if err != nil {
			return nil, fmt.Errorf("read tx op %d: %w", i, err)
		}
		path, err := readTxString(r)
		if err != nil {
			return nil, fmt.Errorf("read tx path %d: %w", i, err)
		}
		aux, err := readTxString(r)
		if err != nil {
			return nil, fmt.Errorf("read tx aux %d: %w", i, err)
		}
		if err := validateLogicalPath(path); err != nil {
			return nil, err
		}
		switch TransactionOpKind(opByte) {
		case TxPut:
			if err := validateStagedPath(aux); err != nil {
				return nil, err
			}
			ops = append(ops, TransactionOp{Kind: TxPut, Path: path, TmpFilePath: filepath.Join(stagingRoot, aux)})
		case TxDelete:
			ops = append(ops, TransactionOp{Kind: TxDelete, Path: path})
		case TxRename:
			if err := validateLogicalPath(aux); err != nil {
				return nil, err
			}
			ops = append(ops, TransactionOp{Kind: TxRename, Path: path, Target: aux})
		default:
			return nil, fmt.Errorf("unknown tx op %d", opByte)
		}
	}
	return ops, nil
}

func readTxString(r *bufio.Reader) (string, error) {
	n, err := binary.ReadUvarint(r)
	if err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("tx string too large: %d", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func validateLogicalPath(path string) error {
	if path == "" || filepath.IsAbs(path) || strings.Contains(path, "\x00") {
		return fmt.Errorf("invalid logical path %q", path)
	}
	clean := filepath.Clean(path)
	if clean == "." || strings.HasPrefix(clean, "..") || strings.HasPrefix(clean, string(filepath.Separator)) {
		return fmt.Errorf("invalid logical path %q", path)
	}
	return nil
}

func validateStagedPath(path string) error {
	if path == "" || filepath.IsAbs(path) || strings.Contains(path, "\x00") {
		return fmt.Errorf("invalid staged path %q", path)
	}
	clean := filepath.Clean(path)
	if clean == "." || strings.HasPrefix(clean, "..") || strings.HasPrefix(clean, string(filepath.Separator)) {
		return fmt.Errorf("invalid staged path %q", path)
	}
	return nil
}
