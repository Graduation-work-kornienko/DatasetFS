//go:build !datasetfs_purego

package pipeline

/*
#cgo pkg-config: libturbojpeg
#include <stdlib.h>
#include <turbojpeg.h>
*/
import "C"

import (
	"fmt"
	"image"
	"unsafe"
)

// libjpegTurbo decodes JPEGs via libjpeg-turbo's TurboJPEG API (tj3*). Per-
// instance state holds an init'd tjhandle + a re-used RGBA decode buffer.
//
// Thread safety: TurboJPEG handles are NOT safe for concurrent use; each
// Decoder goroutine owns exactly one libjpegTurbo via NewDecoder/newJPEGDecoder.
//
// Lifecycle: Close() releases the handle. Decoder.Launch defers Close() so the
// per-session re-init in /initialize_loading cleans up correctly.
//
// Buffer reuse: the RGBA output buffer grows lazily to fit the largest image
// seen so far; subsequent calls reuse it. The buffer is shared across calls
// on the same instance, so the returned image.Image MUST be consumed (resized
// + packed into the caller's scratch) before the next Decode() call. The
// Decoder.decodeSlot loop already follows this invariant.
type libjpegTurbo struct {
	handle C.tjhandle
	buf    []byte // RGBA, 4 bytes per pixel
}

// Decode parses a JPEG and returns an *image.RGBA over the reusable internal
// buffer. Alpha bytes are filled with 255 by TurboJPEG (TJPF_RGBA).
func (d *libjpegTurbo) Decode(b []byte) (image.Image, error) {
	if len(b) == 0 {
		return nil, fmt.Errorf("empty jpeg")
	}
	src := (*C.uchar)(unsafe.Pointer(&b[0]))

	// 1) Read header to get dimensions.
	if C.tj3DecompressHeader(d.handle, src, C.size_t(len(b))) != 0 {
		return nil, d.errFromHandle("tj3DecompressHeader")
	}
	w := int(C.tj3Get(d.handle, C.TJPARAM_JPEGWIDTH))
	h := int(C.tj3Get(d.handle, C.TJPARAM_JPEGHEIGHT))
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid JPEG dimensions %dx%d", w, h)
	}

	// 2) Allocate (or reuse) RGBA output buffer.
	stride := w * 4
	need := h * stride
	if cap(d.buf) < need {
		d.buf = make([]byte, need)
	} else {
		d.buf = d.buf[:need]
	}

	// 3) Decompress directly into the buffer in RGBA byte order.
	rc := C.tj3Decompress8(d.handle, src, C.size_t(len(b)),
		(*C.uchar)(unsafe.Pointer(&d.buf[0])),
		C.int(stride), C.TJPF_RGBA)
	if rc != 0 {
		return nil, d.errFromHandle("tj3Decompress8")
	}

	return &image.RGBA{
		Pix:    d.buf[:need],
		Stride: stride,
		Rect:   image.Rect(0, 0, w, h),
	}, nil
}

// Close destroys the TurboJPEG handle. Safe to call multiple times.
func (d *libjpegTurbo) Close() {
	if d.handle != nil {
		C.tj3Destroy(d.handle)
		d.handle = nil
	}
}

func (d *libjpegTurbo) errFromHandle(op string) error {
	return fmt.Errorf("%s: %s", op, C.GoString(C.tj3GetErrorStr(d.handle)))
}

func newJPEGDecoder() jpegDecoder {
	h := C.tj3Init(C.TJINIT_DECOMPRESS)
	if h == nil {
		// We crash here intentionally: every pipeline construction depends on
		// having a working decoder, and there's no useful fallback (mode is
		// already chosen at /initialize_loading time).
		panic("tj3Init(TJINIT_DECOMPRESS) failed — libjpeg-turbo is broken")
	}
	return &libjpegTurbo{handle: h}
}
