package io

import (
	"time"
)

// Limiter controls the rate of I/O operations

type Limiter struct {
	rate     int64 // bytes per second
	lastTick time.Time
	tokens   int64
}

func NewLimiter(rate int64) *Limiter {
	return &Limiter{
		rate:     rate,
		lastTick: time.Now(),
		tokens:   rate, // Start with full bucket
	}
}

// Wait blocks until n bytes can be transferred
func (l *Limiter) Wait(n int) {
	if l.rate == 0 {
		return // Unlimited
	}

	l.tokens -= int64(n)
	if l.tokens < 0 {
		// Need to wait
		duration := time.Since(l.lastTick)
		refill := int64(duration.Seconds() * float64(l.rate))
		l.tokens += refill

		if l.tokens < 0 {
			// Still negative, need to sleep
			sleep := time.Duration(-l.tokens * int64(time.Second) / l.rate)
			time.Sleep(sleep)
			l.tokens = 0
		}
	}

	l.lastTick = time.Now()
}

// Reader wraps an io.Reader with rate limiting

type LimitedReader struct {
	r       Reader
	limiter *Limiter
}

func NewLimitedReader(r Reader, limiter *Limiter) *LimitedReader {
	return &LimitedReader{r: r, limiter: limiter}
}

func (lr *LimitedReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	if n > 0 {
		lr.limiter.Wait(n)
	}
	return n, err
}

// Writer wraps an io.Writer with rate limiting

type LimitedWriter struct {
	w       Writer
	limiter *Limiter
}

func NewLimitedWriter(w Writer, limiter *Limiter) *LimitedWriter {
	return &LimitedWriter{w: w, limiter: limiter}
}

func (lw *LimitedWriter) Write(p []byte) (n int, err error) {
	n, err = lw.w.Write(p)
	if n > 0 {
		lw.limiter.Wait(n)
	}
	return n, err
}

// Reader is the interface that wraps the basic Read method.
type Reader interface {
	Read(p []byte) (n int, err error)
}

// Writer is the interface that wraps the basic Write method.
type Writer interface {
	Write(p []byte) (n int, err error)
}
