//go:build datasetfs_purego

package pipeline

import (
	"bytes"
	"image"
	"image/jpeg"
)

// pureGoJPEG wraps Go's stdlib image/jpeg decoder. Kept as a baseline for
// thesis A/B against libjpeg-turbo: same code path, only the JPEG step swapped.
//
// Performance note (from optimization 01, iteration 1): pure Go image/jpeg is
// roughly 10 ms per Imagenette JPEG vs ~0.8 ms for libjpeg-based PIL — about
// 12× slower per image. See docs/optimizations/01-server-side-decode.md.
//
// Built only with `-tags datasetfs_purego`. The default build path uses
// decoder_libjpeg.go (libjpeg-turbo via cgo).
type pureGoJPEG struct{}

func (pureGoJPEG) Decode(b []byte) (image.Image, error) {
	return jpeg.Decode(bytes.NewReader(b))
}

func (pureGoJPEG) Close() {}

func newJPEGDecoder() jpegDecoder {
	return pureGoJPEG{}
}
