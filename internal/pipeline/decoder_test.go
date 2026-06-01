package pipeline

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"runtime"
	"strconv"
	"testing"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
	"github.com/Graduation-work-kornienko/DatasetFS/internal/shm"
)

// makeJPEG returns valid JPEG bytes for a solid-color w×h image. Used to seed a
// fake slot buffer without touching disk or the converter.
func makeJPEG(t *testing.T, w, h int, c color.RGBA) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg.Encode: %v", err)
	}
	return buf.Bytes()
}

// buildSlot lays the given per-item byte blobs contiguously into a slot-sized
// buffer for slot 0, returning the buffer and the SlotMeta describing them
// (offsets are global, i.e. include slot 0's base of 0). A nil blob marks an
// item whose bytes are intentionally invalid (decode must fail).
func buildSlot(t *testing.T, blobs [][]byte) ([]byte, *SlotMeta) {
	t.Helper()
	slot := make([]byte, shm.SlotSize)
	var objs []*Metadata
	cursor := int64(0)
	for i, b := range blobs {
		if b == nil {
			// 8 bytes of non-image garbage (no JPEG SOI → image.Decode fails).
			b = []byte{0, 1, 2, 3, 4, 5, 6, 7}
		}
		copy(slot[cursor:cursor+int64(len(b))], b)
		objs = append(objs, &Metadata{
			Metadata: index.Metadata{
				Path:   pathFor(i),
				Offset: cursor, // slot 0 → global offset == local offset
				Size:   int64(len(b)),
			},
			SlotID: 0,
		})
		cursor += int64(len(b))
	}
	return slot, &SlotMeta{Objects: objs, SlotID: 0}
}

func pathFor(i int) string {
	return "img_" + string(rune('A'+i)) + ".jpg"
}

func newTestDecoder(t *testing.T, imageSize, parallelism int) *Decoder {
	t.Helper()
	cfg := DecodeConfig{Mode: DecodeRGBUint8, ImageSize: imageSize, Parallelism: parallelism}
	// nil allocator/channels are fine: decodeSlot takes slotBuf explicitly and
	// we drive the pool directly via startPool/stopPool, never Launch.
	d := NewDecoder(cfg, nil, nil, nil, nil, parallelism)
	d.startPool()
	t.Cleanup(d.stopPool)
	return d
}

// TestDecodeSlot_PreservesOrderAndGaps: a failed item in the middle must be
// dropped while survivors keep original slot order, and every survivor's
// rewritten offset must equal its ORIGINAL index * perItem (fixed, gapped
// layout) — this is what makes the parallel decode deterministic and lock-free.
func TestDecodeSlot_PreservesOrderAndGaps(t *testing.T) {
	const imageSize = 16
	perItem := int64(imageSize * imageSize * 3)
	for _, parallelism := range []int{1, 2, 4} {
		t.Run("K="+strconv.Itoa(parallelism), func(t *testing.T) {
			d := newTestDecoder(t, imageSize, parallelism)

			blobs := [][]byte{
				makeJPEG(t, 32, 32, color.RGBA{255, 0, 0, 255}),
				nil, // index 1 fails to decode
				makeJPEG(t, 24, 24, color.RGBA{0, 255, 0, 255}),
				makeJPEG(t, 40, 40, color.RGBA{0, 0, 255, 255}),
			}
			slot, meta := buildSlot(t, blobs)

			out, used := d.decodeSlot(slot, meta, perItem, imageSize, imageSize)

			if len(out) != 3 {
				t.Fatalf("got %d survivors, want 3", len(out))
			}
			// Order preserved: A, C, D (B at index 1 dropped).
			wantPaths := []string{pathFor(0), pathFor(2), pathFor(3)}
			wantOffsets := []int64{0 * perItem, 2 * perItem, 3 * perItem}
			for i, m := range out {
				if m.Path != wantPaths[i] {
					t.Errorf("survivor %d: path=%s want=%s", i, m.Path, wantPaths[i])
				}
				if m.Offset != wantOffsets[i] {
					t.Errorf("survivor %d (%s): offset=%d want=%d (index*perItem)",
						i, m.Path, m.Offset, wantOffsets[i])
				}
				if m.Size != perItem {
					t.Errorf("survivor %d: size=%d want=%d", i, m.Size, perItem)
				}
			}
			// used spans all dispatched indices (gap included).
			if used != int64(len(blobs))*perItem {
				t.Errorf("used=%d want=%d", used, int64(len(blobs))*perItem)
			}
		})
	}
}

// TestDecodeSlot_AllFail: when every item is garbage, decodeSlot returns no
// survivors so Launch can release the slot.
func TestDecodeSlot_AllFail(t *testing.T) {
	const imageSize = 8
	perItem := int64(imageSize * imageSize * 3)
	d := newTestDecoder(t, imageSize, 2)
	slot, meta := buildSlot(t, [][]byte{nil, nil})
	out, _ := d.decodeSlot(slot, meta, perItem, imageSize, imageSize)
	if len(out) != 0 {
		t.Fatalf("got %d survivors, want 0", len(out))
	}
}

// TestDecodeSlot_PixelsMatchAcrossParallelism: decoding the same slot at K=1 and
// K=4 must produce byte-identical packed output — parallelism must not change
// pixels or layout.
func TestDecodeSlot_PixelsMatchAcrossParallelism(t *testing.T) {
	const imageSize = 16
	perItem := int64(imageSize * imageSize * 3)
	blobs := [][]byte{
		makeJPEG(t, 32, 32, color.RGBA{200, 50, 10, 255}),
		makeJPEG(t, 28, 36, color.RGBA{10, 180, 90, 255}),
		makeJPEG(t, 50, 20, color.RGBA{30, 30, 240, 255}),
	}

	run := func(k int) []byte {
		d := newTestDecoder(t, imageSize, k)
		slot, meta := buildSlot(t, blobs)
		_, used := d.decodeSlot(slot, meta, perItem, imageSize, imageSize)
		// Copy the packed scratch span out (Launch would copy this into the slot).
		got := make([]byte, used)
		copy(got, d.scratch[:used])
		return got
	}

	a := run(1)
	b := run(4)
	if !bytes.Equal(a, b) {
		t.Fatalf("packed output differs between K=1 and K=4 (len %d vs %d)", len(a), len(b))
	}
}

func TestResolveParallelism(t *testing.T) {
	ncpu := runtime.NumCPU()
	cases := []struct {
		requested, numWorkers, want int
	}{
		{4, 8, 4},          // explicit knob wins
		{1, 1, 1},          // explicit 1
		{0, 1, ncpu},       // auto, single pipeline → all cores
		{0, ncpu, 1},       // auto, one pipeline per core → 1 each
		{0, ncpu * 2, 1},   // auto floors at 1 (never 0)
		{-3, 1, ncpu},      // negative treated as auto
	}
	for _, c := range cases {
		if got := resolveParallelism(c.requested, c.numWorkers); got != c.want {
			t.Errorf("resolveParallelism(%d,%d)=%d want=%d",
				c.requested, c.numWorkers, got, c.want)
		}
	}
}
