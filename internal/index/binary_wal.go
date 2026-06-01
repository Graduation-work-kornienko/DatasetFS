package index

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc64"
	"io"
	"os"
	"sync"
)

// Binary WAL format constants
const (
	// Magic number for DatasetFS WAL files
	walMagic uint32 = 0xD5F5574C // 'D','5','F','5','W','A','L' in hex

	// Current version of the binary WAL format
	walVersion uint16 = 1

	// Size of the file header in bytes
	walHeaderSize = 16

	// Size of the record header in bytes (excluding data and checksum)
	recordHeaderSize = 17 // 4 (length) + 1 (op) + 8 (timestamp) + 4 (data length)

	// Size of the record checksum in bytes
	recordChecksumSize = 8
)

// WAL operation types
const (
	opAdd = 1 + iota
	opDelete
	opAppendShard
)

// WALFileHeader represents the header of a binary WAL file
type WALFileHeader struct {
	Magic   uint32 // Magic number (walMagic)
	Version uint16 // Format version
	Flags   uint16 // Reserved for future use
	// Checksum covers Magic, Version, and Flags
	Checksum uint64
}

// WriteTo writes the file header to the writer
func (h *WALFileHeader) WriteTo(w io.Writer) error {
	// Write magic, version, and flags
	if err := binary.Write(w, binary.LittleEndian, h.Magic); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, h.Version); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, h.Flags); err != nil {
		return err
	}

	// Calculate and write checksum
	checksum := crc64.Checksum([]byte{
		byte(h.Magic),
		byte(h.Magic >> 8),
		byte(h.Magic >> 16),
		byte(h.Magic >> 24),
		byte(h.Version),
		byte(h.Version >> 8),
		byte(h.Flags),
		byte(h.Flags >> 8),
	}, crc64.MakeTable(crc64.ISO))

	return binary.Write(w, binary.LittleEndian, checksum)
}

// ReadFrom reads the file header from the reader
func (h *WALFileHeader) ReadFrom(r io.Reader) error {
	// Read magic, version, and flags
	if err := binary.Read(r, binary.LittleEndian, &h.Magic); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Version); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Flags); err != nil {
		return err
	}

	// Read and verify checksum
	var checksum uint64
	if err := binary.Read(r, binary.LittleEndian, &checksum); err != nil {
		return err
	}

	// Calculate expected checksum
	expected := crc64.Checksum([]byte{
		byte(h.Magic),
		byte(h.Magic >> 8),
		byte(h.Magic >> 16),
		byte(h.Magic >> 24),
		byte(h.Version),
		byte(h.Version >> 8),
		byte(h.Flags),
		byte(h.Flags >> 8),
	}, crc64.MakeTable(crc64.ISO))

	if checksum != expected {
		return io.ErrUnexpectedEOF // Using this as a generic error for checksum mismatch
	}

	return nil
}

// RecordHeader represents the header of a WAL record
type RecordHeader struct {
	Length     uint32 // Total length of the record (including header, data, and checksum)
	Op         byte   // Operation type (opAdd, opDelete, opAppendShard)
	Timestamp  int64  // Unix timestamp
	DataLength uint32 // Length of the data field
	// Checksum covers all fields above (Length, Op, Timestamp, DataLength) and the data
	Checksum uint64
}

// WriteTo writes the record header to the writer
func (h *RecordHeader) WriteTo(w io.Writer) error {
	// Write length, op, timestamp, and data length
	if err := binary.Write(w, binary.LittleEndian, h.Length); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, h.Op); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, h.Timestamp); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, h.DataLength); err != nil {
		return err
	}

	// Write checksum
	return binary.Write(w, binary.LittleEndian, h.Checksum)
}

// ReadFrom reads the record header from the reader
func (h *RecordHeader) ReadFrom(r io.Reader) error {
	// Read length, op, timestamp, and data length
	if err := binary.Read(r, binary.LittleEndian, &h.Length); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Op); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.Timestamp); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &h.DataLength); err != nil {
		return err
	}

	// Read checksum
	return binary.Read(r, binary.LittleEndian, &h.Checksum)
}

// writeVarint writes a variable-length integer to the writer
func writeVarint(w io.Writer, i int) error {
	var buf [10]byte
	n := binary.PutUvarint(buf[:], uint64(i))
	_, err := w.Write(buf[:n])
	return err
}

// readVarint reads a variable-length integer from the reader
func readVarint(r io.Reader) (int, error) {
	var buf [1]byte
	var result uint64
	var shift uint
	for {
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			return 0, err
		}
		b := buf[0]
		result |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			break
		}
		shift += 7
		if shift > 63 {
			return 0, io.ErrUnexpectedEOF
		}
	}
	return int(result), nil
}

// writeString writes a string with its length as a varint prefix
func writeString(w io.Writer, s string) error {
	if err := writeVarint(w, len(s)); err != nil {
		return err
	}
	_, err := w.Write([]byte(s))
	return err
}

// readString reads a string with its length as a varint prefix
func readString(r io.Reader) (string, error) {
	length, err := readVarint(r)
	if err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// BinaryWAL provides binary format WAL operations.
//
// All public methods that touch the file are serialized by mu, mirroring
// JSONWAL: MutationManager already serializes its own callers, but the WAL
// must be safe on its own so a future concurrent caller can't interleave a
// record header with another record's data and corrupt the log.
type BinaryWAL struct {
	mu   sync.Mutex
	file *os.File
	// Reusable buffer for calculating checksums
	checksumBuf []byte
}

// NewBinaryWAL creates a new binary WAL writer
func NewBinaryWAL(file *os.File) (*BinaryWAL, error) {
	wal := &BinaryWAL{
		file: file,
		// Pre-allocate buffer for checksum calculations
		checksumBuf: make([]byte, 1024),
	}

	// Write file header
	header := &WALFileHeader{
		Magic:   walMagic,
		Version: walVersion,
		Flags:   0,
	}
	if err := header.WriteTo(file); err != nil {
		return nil, err
	}

	return wal, nil
}

// OpenBinaryWAL opens an existing binary WAL file for reading and replay
func OpenBinaryWAL(file *os.File) (*BinaryWAL, error) {
	wal := &BinaryWAL{
		file: file,
		// Pre-allocate buffer for checksum calculations
		checksumBuf: make([]byte, 1024),
	}

	// Read and verify file header
	header := &WALFileHeader{}
	if err := header.ReadFrom(file); err != nil {
		return nil, fmt.Errorf("read WAL file header: %w", err)
	}

	// Validate magic number and version
	if header.Magic != walMagic {
		return nil, fmt.Errorf("invalid magic number: got 0x%x, expected 0x%x", header.Magic, walMagic)
	}
	if header.Version != walVersion {
		return nil, fmt.Errorf("unsupported version: got %d, expected %d", header.Version, walVersion)
	}

	return wal, nil
}

// ReadEntry reads a single WAL entry from the binary log
func (bw *BinaryWAL) ReadEntry() (*WALEntry, error) {
	// Read record header
	header := &RecordHeader{}
	if err := header.ReadFrom(bw.file); err != nil {
		return nil, err
	}

	// Read data payload
	data := make([]byte, header.DataLength)
	if _, err := io.ReadFull(bw.file, data); err != nil {
		return nil, err
	}

	// Read record checksum
	var recordChecksum uint64
	if err := binary.Read(bw.file, binary.LittleEndian, &recordChecksum); err != nil {
		return nil, err
	}

	// Verify checksum
	// Create buffer with header fields and data for checksum calculation
	if len(bw.checksumBuf) < recordHeaderSize+len(data) {
		bw.checksumBuf = make([]byte, recordHeaderSize+len(data))
	}
	// Pack header fields into buffer
	binary.LittleEndian.PutUint32(bw.checksumBuf[0:4], header.Length)
	bw.checksumBuf[4] = header.Op
	binary.LittleEndian.PutUint64(bw.checksumBuf[5:13], uint64(header.Timestamp))
	binary.LittleEndian.PutUint32(bw.checksumBuf[13:17], header.DataLength)
	// Copy data after header
	copy(bw.checksumBuf[recordHeaderSize:], data)

	calculatedChecksum := crc64.Checksum(bw.checksumBuf[:recordHeaderSize+len(data)], crc64.MakeTable(crc64.ISO))
	if recordChecksum != calculatedChecksum {
		return nil, fmt.Errorf("record checksum mismatch: got 0x%x, expected 0x%x", recordChecksum, calculatedChecksum)
	}

	// Deserialize data based on operation type
	e := &WALEntry{Timestamp: header.Timestamp}
	switch header.Op {
	case opAdd:
		meta, err := readAddData(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("read add data: %w", err)
		}
		e.Op = OpAdd
		e.Add = meta
	case opDelete:
		path, err := readDeleteData(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("read delete data: %w", err)
		}
		e.Op = OpDelete
		e.Delete = path
	case opAppendShard:
		shard, err := readShardData(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("read shard data: %w", err)
		}
		e.Op = OpAppendShard
		e.Shard = shard
	default:
		return nil, fmt.Errorf("unknown operation code: %d", header.Op)
	}

	return e, nil
}

// Replay reads the binary WAL from start to EOF and applies every record to idx.
// On parse error or apply failure it returns immediately.
// Restores the file position to EOF on exit so subsequent operations can continue appending.
func (bw *BinaryWAL) Replay(idx *CoreIndex) (applied int, err error) {
	bw.mu.Lock()
	defer bw.mu.Unlock()

	if bw.file == nil {
		return 0, fmt.Errorf("wal closed")
	}

	// Seek to beginning of file
	if _, err := bw.file.Seek(0, io.SeekStart); err != nil {
		return 0, fmt.Errorf("seek to start: %w", err)
	}

	// Read and verify file header
	header := &WALFileHeader{}
	if err := header.ReadFrom(bw.file); err != nil {
		return 0, fmt.Errorf("read WAL file header: %w", err)
	}

	// Validate magic number and version
	if header.Magic != walMagic {
		return 0, fmt.Errorf("invalid magic number: got 0x%x, expected 0x%x", header.Magic, walMagic)
	}
	if header.Version != walVersion {
		return 0, fmt.Errorf("unsupported version: got %d, expected %d", header.Version, walVersion)
	}

	// Process records until EOF
	for {
		e, err := bw.ReadEntry()
		if err != nil {
			if err == io.EOF {
				break // Normal end of file
			}
			return applied, fmt.Errorf("read WAL entry %d: %w", applied+1, err)
		}

		// Apply the entry to the index
		if err := applyEntry(e, idx); err != nil {
			return applied, fmt.Errorf("apply WAL entry %d (op=%s): %w", applied+1, e.Op, err)
		}
		applied++
	}

	// Restore file position to end for potential future writes
	if _, err := bw.file.Seek(0, io.SeekEnd); err != nil {
		// Log error but don't fail the replay operation
		fmt.Fprintf(os.Stderr, "warning: failed to restore file position: %v\n", err)
	}

	return applied, nil
}

// WriteEntry writes a WAL entry in binary format
func (bw *BinaryWAL) WriteEntry(e *WALEntry) error {
	bw.mu.Lock()
	defer bw.mu.Unlock()

	if bw.file == nil {
		return fmt.Errorf("wal closed")
	}

	// Calculate data length and serialize data
	var data []byte
	var err error
	switch e.Op {
	case OpAdd:
		data, err = serializeAddData(e.Add)
	case OpDelete:
		data, err = serializeDeleteData(e.Delete)
	case OpAppendShard:
		data, err = serializeShardData(e.Shard)
	default:
		return fmt.Errorf("unknown operation: %v", e.Op)
	}
	if err != nil {
		return err
	}

	// Calculate total record length
	recordLength := recordHeaderSize + uint32(len(data)) + recordChecksumSize

	// Calculate checksum
	headerBuf := make([]byte, recordHeaderSize)
	binary.LittleEndian.PutUint32(headerBuf[0:4], recordLength)
	headerBuf[4] = getOpCode(e.Op)
	binary.LittleEndian.PutUint64(headerBuf[5:13], uint64(e.Timestamp))
	binary.LittleEndian.PutUint32(headerBuf[13:17], uint32(len(data)))

	// Resize checksum buffer if needed
	if len(bw.checksumBuf) < len(headerBuf)+len(data) {
		bw.checksumBuf = make([]byte, len(headerBuf)+len(data))
	}
	copy(bw.checksumBuf, headerBuf)
	copy(bw.checksumBuf[len(headerBuf):], data)
	checksum := crc64.Checksum(bw.checksumBuf[:len(headerBuf)+len(data)], crc64.MakeTable(crc64.ISO))

	// Write record header
	header := &RecordHeader{
		Length:     recordLength,
		Op:         getOpCode(e.Op),
		Timestamp:  e.Timestamp,
		DataLength: uint32(len(data)),
		Checksum:   checksum,
	}
	if err := header.WriteTo(bw.file); err != nil {
		return err
	}

	// Write data
	if _, err := bw.file.Write(data); err != nil {
		return err
	}

	// Write checksum
	if err := binary.Write(bw.file, binary.LittleEndian, checksum); err != nil {
		return err
	}

	// fsync for durability
	return bw.file.Sync()
}

// getOpCode converts WALOp to binary operation code
func getOpCode(op WALOp) byte {
	switch op {
	case OpAdd:
		return opAdd
	case OpDelete:
		return opDelete
	case OpAppendShard:
		return opAppendShard
	default:
		return 0
	}
}

// serializeAddData serializes Add operation data
func serializeAddData(meta *Metadata) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeAddData(&buf, meta); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// serializeDeleteData serializes Delete operation data
func serializeDeleteData(path string) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeDeleteData(&buf, path); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// serializeShardData serializes AppendShard operation data
func serializeShardData(shard *Shard) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeShardData(&buf, shard); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// writeAddData writes the data for an Add operation
func writeAddData(w io.Writer, meta *Metadata) error {
	// Write path
	if err := writeString(w, meta.Path); err != nil {
		return err
	}
	// Write shard ID
	if err := binary.Write(w, binary.LittleEndian, int32(meta.ShardID)); err != nil {
		return err
	}
	// Write offset
	if err := binary.Write(w, binary.LittleEndian, meta.Offset); err != nil {
		return err
	}
	// Write size
	if err := binary.Write(w, binary.LittleEndian, meta.Size); err != nil {
		return err
	}
	// Write object metadata
	if meta.ObjectMetadata == nil {
		if err := writeVarint(w, 0); err != nil {
			return err
		}
	} else {
		if err := writeVarint(w, len(meta.ObjectMetadata)); err != nil {
			return err
		}
		if _, err := w.Write(meta.ObjectMetadata); err != nil {
			return err
		}
	}
	return nil
}

// readAddData reads the data for an Add operation
func readAddData(r io.Reader) (*Metadata, error) {
	meta := &Metadata{}

	// Read path
	var err error
	meta.Path, err = readString(r)
	if err != nil {
		return nil, err
	}

	// Read shard ID
	var shardID int32
	if err := binary.Read(r, binary.LittleEndian, &shardID); err != nil {
		return nil, err
	}
	meta.ShardID = int(shardID)

	// Read offset
	if err := binary.Read(r, binary.LittleEndian, &meta.Offset); err != nil {
		return nil, err
	}

	// Read size
	if err := binary.Read(r, binary.LittleEndian, &meta.Size); err != nil {
		return nil, err
	}

	// Read object metadata
	length, err := readVarint(r)
	if err != nil {
		return nil, err
	}
	if length > 0 {
		meta.ObjectMetadata = make([]byte, length)
		if _, err := io.ReadFull(r, meta.ObjectMetadata); err != nil {
			return nil, err
		}
	}

	return meta, nil
}

// writeDeleteData writes the data for a Delete operation
func writeDeleteData(w io.Writer, path string) error {
	return writeString(w, path)
}

// readDeleteData reads the data for a Delete operation
func readDeleteData(r io.Reader) (string, error) {
	return readString(r)
}

// writeShardData writes the data for an AppendShard operation
func writeShardData(w io.Writer, shard *Shard) error {
	// Write shard number
	if err := binary.Write(w, binary.LittleEndian, int32(shard.Number)); err != nil {
		return err
	}
	// Write shard type
	if err := writeString(w, string(shard.Type)); err != nil {
		return err
	}
	// Write total size
	if err := binary.Write(w, binary.LittleEndian, shard.TotalSize); err != nil {
		return err
	}
	// Write objects count
	if err := writeVarint(w, len(shard.Objects)); err != nil {
		return err
	}
	// Write each object
	for _, obj := range shard.Objects {
		if err := writeAddData(w, obj); err != nil {
			return err
		}
	}
	return nil
}

// readShardData reads the data for an AppendShard operation
func readShardData(r io.Reader) (*Shard, error) {
	shard := &Shard{}

	// Read shard number
	var number int32
	if err := binary.Read(r, binary.LittleEndian, &number); err != nil {
		return nil, err
	}
	shard.Number = int(number)

	// Read shard type
	shardType, err := readString(r)
	if err != nil {
		return nil, err
	}
	shard.Type = ShardType(shardType)

	// Read total size
	if err := binary.Read(r, binary.LittleEndian, &shard.TotalSize); err != nil {
		return nil, err
	}

	// Read objects count
	count, err := readVarint(r)
	if err != nil {
		return nil, err
	}

	// Read each object
	shard.Objects = make([]*Metadata, count)
	for i := 0; i < count; i++ {
		obj, err := readAddData(r)
		if err != nil {
			return nil, err
		}
		shard.Objects[i] = obj
	}

	return shard, nil
}

// Path returns the absolute path of the WAL file. Useful for diagnostics.
func (bw *BinaryWAL) Path() string {
	return bw.file.Name()
}

// LogAdd records an AddFile mutation. Caller MUST call this before
// CoreIndex.AddFile so a crash between them is replay-recoverable.
func (bw *BinaryWAL) LogAdd(meta *Metadata) error {
	if meta == nil {
		return fmt.Errorf("LogAdd: meta is nil")
	}
	return bw.WriteEntry(&WALEntry{Op: OpAdd, Add: meta})
}

// LogDelete records a tombstone mutation.
func (bw *BinaryWAL) LogDelete(path string) error {
	if path == "" {
		return fmt.Errorf("LogDelete: empty path")
	}
	return bw.WriteEntry(&WALEntry{Op: OpDelete, Delete: path})
}

// LogAppendShard records an entire shard's metadata (including its Objects).
// Used by dataset-init paths that bulk-append shards.
func (bw *BinaryWAL) LogAppendShard(shard *Shard) error {
	if shard == nil {
		return fmt.Errorf("LogAppendShard: shard is nil")
	}
	return bw.WriteEntry(&WALEntry{Op: OpAppendShard, Shard: shard})
}

// Truncate empties the WAL. Call ONLY after the manifest has been durably
// stored — truncating before that would lose the very mutations the manifest
// was supposed to capture.
func (bw *BinaryWAL) Truncate() error {
	bw.mu.Lock()
	defer bw.mu.Unlock()

	if bw.file == nil {
		return fmt.Errorf("wal closed")
	}

	if err := bw.file.Truncate(0); err != nil {
		return fmt.Errorf("truncate wal: %w", err)
	}
	if _, err := bw.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek wal start: %w", err)
	}
	// A binary WAL must always begin with the file header; truncating to 0
	// erased it, so rewrite it to keep the file (and this handle) valid for
	// subsequent appends and replays.
	header := &WALFileHeader{Magic: walMagic, Version: walVersion, Flags: 0}
	if err := header.WriteTo(bw.file); err != nil {
		return fmt.Errorf("rewrite wal header: %w", err)
	}
	return bw.file.Sync()
}

// Close flushes and releases the WAL file descriptor.
func (bw *BinaryWAL) Close() error {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	if bw.file == nil {
		return nil
	}
	err := bw.file.Close()
	bw.file = nil
	return err
}
