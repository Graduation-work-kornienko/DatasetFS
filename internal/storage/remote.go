package storage

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
)

// RemoteStorage handles downloading and caching of remote files
type RemoteStorage struct {
	CacheDir string
	client   *http.Client
	mu       sync.Mutex
	cache    map[string]string // URL to local path
}

// NewRemoteStorage creates a new RemoteStorage with the specified cache directory
func NewRemoteStorage(cacheDir string) *RemoteStorage {
	return &RemoteStorage{
		CacheDir: cacheDir,
		client:   &http.Client{},
		cache:    make(map[string]string),
	}
}

// Download downloads a file from a URL and returns the local path
func (rs *RemoteStorage) Download(ctx context.Context, url string) (string, error) {
	rs.mu.Lock()
	cachedPath, exists := rs.cache[url]
	rs.mu.Unlock()

	if exists {
		return cachedPath, nil
	}

	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(rs.CacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Create a temporary file
	tmpFile, err := os.CreateTemp(rs.CacheDir, "download-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tmpFile.Close()

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Execute request
	resp, err := rs.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Copy response body to file
	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		return "", fmt.Errorf("failed to copy response body: %w", err)
	}

	// Close the file before renaming
	if err := tmpFile.Close(); err != nil {
		return "", fmt.Errorf("failed to close temp file: %w", err)
	}

	// Generate final path based on URL
	filename := filepath.Base(url)
	if filename == "." || filename == "/" {
		filename = "index"
	}
	finalPath := filepath.Join(rs.CacheDir, filename)

	// Rename temp file to final path
	if err := os.Rename(tmpFile.Name(), finalPath); err != nil {
		// If rename fails, try to copy and remove
		if copyErr := copyFile(tmpFile.Name(), finalPath); copyErr != nil {
			return "", fmt.Errorf("failed to rename temp file: %w", err)
		}
		if removeErr := os.Remove(tmpFile.Name()); removeErr != nil {
			// Log the error but continue
			fmt.Printf("warning: failed to remove temp file %s: %v\n", tmpFile.Name(), removeErr)
		}
	}

	// Cache the result
	rs.mu.Lock()
	rs.cache[url] = finalPath
	rs.mu.Unlock()

	return finalPath, nil
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// GetLocalPath returns the local path for a URL, downloading it if necessary
func (rs *RemoteStorage) GetLocalPath(ctx context.Context, url string) (string, error) {
	// Check if it's already a local path
	if !isURL(url) {
		return url, nil
	}

	// Download the file if it's a URL
	return rs.Download(ctx, url)
}

// isURL checks if a string is a URL
func isURL(s string) bool {
	return len(s) > 7 && (s[:7] == "http://" || s[:8] == "https://")
}

// IsURL checks if a string is a URL (public function)
func IsURL(s string) bool {
	return isURL(s)
}

// DownloadStream downloads a file from a URL and returns the response body for streaming
func (rs *RemoteStorage) DownloadStream(ctx context.Context, url string) (*http.Response, error) {
	rs.mu.Lock()
	cachedPath, exists := rs.cache[url]
	rs.mu.Unlock()

	if exists {
		// File is already cached, return a file reader
		file, err := os.Open(cachedPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open cached file: %w", err)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       file,
			Header:     make(http.Header),
		}, nil
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Execute request
	resp, err := rs.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download %s: %w", url, err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Return the response body for streaming
	return resp, nil
}
