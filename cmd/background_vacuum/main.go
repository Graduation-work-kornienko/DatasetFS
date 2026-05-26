package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/Graduation-work-kornienko/DatasetFS/internal/index"
)

// ResourceMonitor monitors system resources and adjusts vacuumer behavior
// based on current system load
//
// The monitor checks CPU, memory, and I/O usage to determine if the system
// is under heavy load, and adjusts the vacuumer's throttling accordingly.
// This ensures the vacuumer doesn't impact training workloads.
type ResourceMonitor struct {
	config *Config
}

// NewResourceMonitor creates a new resource monitor
func NewResourceMonitor(config *Config) *ResourceMonitor {
	return &ResourceMonitor{config: config}
}

// ShouldThrottle checks if the vacuumer should throttle its operations
// based on current system load
func (rm *ResourceMonitor) ShouldThrottle() bool {
	// Check if we should always throttle (config override)
	if rm.config.Throttle > 0 {
		return true
	}

	// Check system load - if any metric is high, throttle
	if rm.isCPULoadHigh() || rm.isMemoryUsageHigh() || rm.isIOLoadHigh() {
		return true
	}

	return false
}

// GetRecommendedThrottle returns the recommended throttle rate based on
// current system conditions. Returns 0 for unlimited if system is idle.
func (rm *ResourceMonitor) GetRecommendedThrottle() int64 {
	// If system is under heavy load, return conservative throttle
	if rm.ShouldThrottle() {
		return 10 * 1024 * 1024 // 10MB/s when system is busy
	}

	// Otherwise, allow higher throughput
	return 50 * 1024 * 1024 // 50MB/s when system is idle
}

// isCPULoadHigh checks if CPU load is high
// This is a simplified implementation - in practice, you might want to use
// more sophisticated methods to determine CPU load
func (rm *ResourceMonitor) isCPULoadHigh() bool {
	// For now, check if we're running on a system with high CPU count
	// and assume load is high if we're using a lot of CPU
	// In a real implementation, you would read from /proc/loadavg or use
	// runtime.NumGoroutine() and other metrics
	return runtime.NumGoroutine() > 1000
}

// isMemoryUsageHigh checks if memory usage is high
// This is a simplified implementation - in practice, you might want to use
// more sophisticated methods to determine memory usage
func (rm *ResourceMonitor) isMemoryUsageHigh() bool {
	// For now, just return false
	// In a real implementation, you would check system memory usage
	// via runtime.MemStats or system calls
	return false
}

// isIOLoadHigh checks if I/O load is high
// This is a simplified implementation - in practice, you might want to use
// more sophisticated methods to determine I/O load
func (rm *ResourceMonitor) isIOLoadHigh() bool {
	// For now, just return false
	// In a real implementation, you would monitor disk I/O statistics
	// from /proc/diskstats or similar
	return false
}

// Config represents the configuration for the background vacuumer
type Config struct {
	RootPath               string        `json:"root_path"`
	CheckInterval          time.Duration `json:"check_interval"`
	MaxShardSize           int64         `json:"max_shard_size"`
	PreserveWAL            bool          `json:"preserve_wal"`
	Throttle               int64         `json:"throttle"`
	FragmentationThreshold float64       `json:"fragmentation_threshold"`
	MinFreeSpace           int64         `json:"min_free_space"`
	Schedule               string        `json:"schedule"`
}

// DefaultConfig returns the default configuration for the background vacuumer
func DefaultConfig() *Config {
	return &Config{
		CheckInterval:          time.Minute * 5,
		MaxShardSize:           100 * 1024 * 1024, // 100MB
		PreserveWAL:            false,
		Throttle:               10 * 1024 * 1024,  // 10MB/s
		FragmentationThreshold: 0.3,               // 30% fragmentation
		MinFreeSpace:           100 * 1024 * 1024, // 100MB
		Schedule:               "",                // Run continuously
	}
}

// State represents the persistent state of the background vacuumer
type State struct {
	LastRunTime        time.Time     `json:"last_run_time"`
	LastRunDuration    time.Duration `json:"last_run_duration"`
	LastRunFiles       int           `json:"last_run_files"`
	LastRunShards      int           `json:"last_run_shards"`
	Paused             bool          `json:"paused"`
	Progress           float64       `json:"progress"`
	CurrentPhase       string        `json:"current_phase"`
	LastCheckpointTime time.Time     `json:"last_checkpoint_time"`
}

// BackgroundVacuumer manages the background vacuum operations
type BackgroundVacuumer struct {
	config  *Config
	state   *State
	ctx     context.Context
	cancel  context.CancelFunc
	monitor *ResourceMonitor
}

// NewBackgroundVacuumer creates a new background vacuumer instance
func NewBackgroundVacuumer(config *Config) *BackgroundVacuumer {
	ctx, cancel := context.WithCancel(context.Background())
	return &BackgroundVacuumer{
		config:  config,
		state:   &State{},
		ctx:     ctx,
		cancel:  cancel,
		monitor: NewResourceMonitor(config),
	}
}

// Start begins the background vacuumer service
func (bv *BackgroundVacuumer) Start() error {
	log.Printf("Starting background vacuumer with config: %+v", bv.config)

	// Load or initialize state
	if err := bv.loadState(); err != nil {
		log.Printf("Failed to load state, starting fresh: %v", err)
	}

	// Start the main loop
	go bv.run()

	return nil
}

// Stop terminates the background vacuumer service
func (bv *BackgroundVacuumer) Stop() {
	log.Println("Stopping background vacuumer")
	bv.cancel()
	bv.saveState() // Best effort save
}

// run is the main loop for the background vacuumer
func (bv *BackgroundVacuumer) run() {
	// Create ticker for periodic checks
	ticker := time.NewTicker(bv.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-bv.ctx.Done():
			log.Println("Background vacuumer context cancelled")
			return
		case <-ticker.C:
			// Check if vacuumer should run
			if bv.state.Paused {
				log.Println("Background vacuumer is paused")
				continue
			}

			// Run vacuum check
			if shouldRun, reason := bv.shouldRunVacuum(); shouldRun {
				log.Printf("Running vacuum: %s", reason)
				start := time.Now()
				if err := bv.runVacuum(); err != nil {
					log.Printf("Vacuum failed: %v", err)
				} else {
					bv.state.LastRunDuration = time.Since(start)
					log.Printf("Vacuum completed in %v", bv.state.LastRunDuration)
				}
				bv.state.LastRunTime = time.Now()
				bv.saveState()
			} else {
				log.Printf("Vacuum not needed: %s", reason)
			}
		}
	}
}

// shouldRunVacuum determines if vacuum should be run based on current conditions
func (bv *BackgroundVacuumer) shouldRunVacuum() (bool, string) {
	// Check if root path is configured
	if bv.config.RootPath == "" {
		return false, "root path not configured"
	}

	// Check if root path exists
	if _, err := os.Stat(bv.config.RootPath); os.IsNotExist(err) {
		return false, "root path does not exist"
	}

	// Check for sufficient free space
	if !bv.hasSufficientFreeSpace() {
		return false, "insufficient free space"
	}

	// Check if DatasetFS daemon is running
	if bv.isDaemonRunning() {
		return false, "DatasetFS daemon is running"
	}

	// Check if there are active mutations
	if bv.hasActiveMutations() {
		return false, "active mutations in progress"
	}

	// Check fragmentation level
	fragmentation, err := bv.calculateFragmentation()
	if err != nil {
		return false, fmt.Sprintf("failed to calculate fragmentation: %v", err)
	}

	if fragmentation < bv.config.FragmentationThreshold {
		return false, fmt.Sprintf("fragmentation %.2f%% below threshold %.2f%%", fragmentation*100, bv.config.FragmentationThreshold*100)
	}

	// Check schedule if specified
	if bv.config.Schedule != "" {
		if !bv.matchesSchedule() {
			return false, "current time does not match schedule"
		}
	}

	return true, fmt.Sprintf("fragmentation %.2f%% exceeds threshold %.2f%%", fragmentation*100, bv.config.FragmentationThreshold*100)
}

// isDaemonRunning checks if the DatasetFS daemon is currently running
func (bv *BackgroundVacuumer) isDaemonRunning() bool {
	// Implementation would check for daemon lock files or process status
	// This is a placeholder for the actual implementation
	lockPath := filepath.Join(bv.config.RootPath, ".daemon_lock")
	_, err := os.Stat(lockPath)
	return !os.IsNotExist(err)
}

// hasActiveMutations checks if there are active mutations in the WAL
func (bv *BackgroundVacuumer) hasActiveMutations() bool {
	// Load manifest
	manifest := &index.Manifest{Root: bv.config.RootPath}
	if err := manifest.Load(nil); err != nil {
		log.Printf("Failed to load manifest: %v", err)
		return true // Assume there are mutations if we can't load
	}

	// Open WAL to check its size
	walPath := filepath.Join(bv.config.RootPath, "wal.bin")
	if fileInfo, err := os.Stat(walPath); err == nil {
		// If WAL is larger than a threshold, assume there are active mutations
		// This is a simplified check - in practice, you might want to check
		// the actual content or use a more sophisticated method
		return fileInfo.Size() > 1024*1024 // 1MB threshold
	}

	return false
}

// hasSufficientFreeSpace checks if there is enough free space for vacuum operation
func (bv *BackgroundVacuumer) hasSufficientFreeSpace() bool {
	if bv.config.MinFreeSpace == 0 {
		return true // No minimum free space requirement
	}

	// Get free space on the filesystem
	// This is a simplified implementation - in practice, you might want to use
	// syscall.Statfs or similar for more accurate results
	// For now, we'll check if we can create a temporary file of the minimum size
	tempFile := filepath.Join(bv.config.RootPath, ".vacuum_check.tmp")
	file, err := os.Create(tempFile)
	if err != nil {
		log.Printf("Failed to create temp file for space check: %v", err)
		return false
	}
	defer func() {
		file.Close()
		os.Remove(tempFile)
	}()

	// Try to write the minimum free space amount
	buf := make([]byte, 1024*1024) // 1MB buffer
	remaining := bv.config.MinFreeSpace
	for remaining > 0 {
		writeSize := int64(len(buf))
		if writeSize > remaining {
			writeSize = remaining
		}
		if _, err := file.Write(buf[:writeSize]); err != nil {
			log.Printf("Failed to write to temp file for space check: %v", err)
			return false
		}
		remaining -= writeSize
	}

	return true
}

// calculateFragmentation calculates the current fragmentation level
func (bv *BackgroundVacuumer) calculateFragmentation() (float64, error) {
	// Load manifest
	manifest := &index.Manifest{Root: bv.config.RootPath}
	if err := manifest.Load(nil); err != nil {
		return 0, fmt.Errorf("failed to load manifest: %w", err)
	}

	// Calculate total logical size (sum of all file sizes)
	var totalLogicalSize int64
	var deletedFiles int
	for _, meta := range manifest.Files {
		if meta.Deleted {
			deletedFiles++
		} else {
			totalLogicalSize += meta.Size
		}
	}

	// Calculate total physical size (sum of all shard sizes)
	var totalPhysicalSize int64
	for _, shard := range manifest.ShardsMeta {
		totalPhysicalSize += shard.TotalSize
	}

	// If there's no physical size, return 0
	if totalPhysicalSize == 0 {
		return 0, nil
	}

	// Calculate fragmentation as the ratio of wasted space to physical size
	// Wasted space = physical size - logical size
	wastedSpace := totalPhysicalSize - totalLogicalSize
	fragmentation := float64(wastedSpace) / float64(totalPhysicalSize)

	log.Printf("Fragmentation: %.2f%% (%d bytes wasted of %d bytes physical, %d deleted files)",
		fragmentation*100, wastedSpace, totalPhysicalSize, deletedFiles)

	return fragmentation, nil
}

// matchesSchedule checks if the current time matches the configured schedule
// This is a placeholder - in a real implementation, you would parse cron expressions
func (bv *BackgroundVacuumer) matchesSchedule() bool {
	// For now, just return true
	// In a complete implementation, this would parse cron expressions
	// and check if the current time matches the schedule
	return true
}

// runVacuum executes the vacuum operation
func (bv *BackgroundVacuumer) runVacuum() error {
	// Reuse the existing vacuum command with background parameters
	// This ensures consistency with the manual vacuum operation

	// Determine throttle rate based on system load
	throttle := bv.config.Throttle
	if throttle == 0 {
		// If no explicit throttle, use recommended based on system load
		throttle = bv.monitor.GetRecommendedThrottle()
	}

	// Update state to indicate vacuum is running
	bv.state.CurrentPhase = "vacuum"
	bv.state.Progress = 0.0
	bv.state.LastCheckpointTime = time.Now()
	bv.saveState()

	// Create a temporary config file for the vacuum command
	// In a real implementation, we would call the vacuum package directly
	// For now, we'll execute the vacuum command as a subprocess

	cmd := exec.Command("go", "run", "cmd/vacuum/main.go",
		"--root", bv.config.RootPath,
		"--max-shard-size", fmt.Sprintf("%d", bv.config.MaxShardSize),
		"--preserve-wal", fmt.Sprintf("%t", bv.config.PreserveWAL),
		"--background", "true",
		"--throttle", fmt.Sprintf("%d", throttle),
		"--verbose")

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	log.Printf("Executing vacuum command: %s", cmd.String())
	if err := cmd.Run(); err != nil {
		log.Printf("Vacuum command failed: %v\nStdout: %s\nStderr: %s", err, stdout.String(), stderr.String())
		return fmt.Errorf("vacuum command failed: %w", err)
	}

	log.Printf("Vacuum command succeeded\nStdout: %s", stdout.String())

	// Parse output to update state
	// This is a simplified implementation
	// In a real implementation, you would parse the output to get the number of files and shards
	bv.state.LastRunFiles = 0  // Placeholder
	bv.state.LastRunShards = 0 // Placeholder
	bv.state.Progress = 1.0
	bv.state.CurrentPhase = "idle"
	bv.saveState()

	return nil
}

// loadState loads the persistent state from disk
func (bv *BackgroundVacuumer) loadState() error {
	statePath := filepath.Join(bv.config.RootPath, ".vacuum_state.json")
	data, err := os.ReadFile(statePath)
	if err != nil {
		if os.IsNotExist(err) {
			// No state file, that's ok
			return nil
		}
		return fmt.Errorf("failed to read state file: %w", err)
	}

	if err := json.Unmarshal(data, bv.state); err != nil {
		return fmt.Errorf("failed to unmarshal state: %w", err)
	}

	log.Printf("Loaded state: %+v", bv.state)
	return nil
}

// saveState saves the persistent state to disk
func (bv *BackgroundVacuumer) saveState() {
	if bv.config.RootPath == "" {
		return // Can't save without root path
	}

	statePath := filepath.Join(bv.config.RootPath, ".vacuum_state.json")
	data, err := json.MarshalIndent(bv.state, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal state: %v", err)
		return
	}

	// Write to temporary file first for atomic update
	tempPath := statePath + ".tmp"
	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		log.Printf("Failed to write temporary state file: %v", err)
		return
	}

	// Atomic rename to final location
	if err := os.Rename(tempPath, statePath); err != nil {
		log.Printf("Failed to rename state file: %v", err)
		return
	}

	log.Printf("Saved state to %s", statePath)
}

func main() {
	var configPath string
	var rootPath string
	var checkInterval int
	var maxShardSize int64
	var preserveWAL bool
	var throttle int64
	var fragmentationThreshold float64
	var minFreeSpace int64
	var schedule string

	flag.StringVar(&configPath, "config", "", "Path to configuration file")
	flag.StringVar(&rootPath, "root", "", "Path to DatasetFS root directory")
	flag.IntVar(&checkInterval, "check-interval", 300, "Check interval in seconds")
	flag.Int64Var(&maxShardSize, "max-shard-size", 100*1024*1024, "Maximum size for output shards in bytes")
	flag.BoolVar(&preserveWAL, "preserve-wal", false, "Preserve WAL after vacuum")
	flag.Int64Var(&throttle, "throttle", 10*1024*1024, "Throttle disk bandwidth in bytes/sec")
	flag.Float64Var(&fragmentationThreshold, "fragmentation-threshold", 0.3, "Fragmentation threshold (0.0-1.0)")
	flag.Int64Var(&minFreeSpace, "min-free-space", 100*1024*1024, "Minimum free space required in bytes")
	flag.StringVar(&schedule, "schedule", "", "Schedule for vacuum (cron format)")
	flag.Parse()

	// Load config from file if specified
	config := DefaultConfig()
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			log.Fatalf("Failed to read config file: %v", err)
		}

		if err := json.Unmarshal(data, config); err != nil {
			log.Fatalf("Failed to parse config file: %v", err)
		}
	}

	// Override with command line flags
	if rootPath != "" {
		config.RootPath = rootPath
	}
	if checkInterval > 0 {
		config.CheckInterval = time.Duration(checkInterval) * time.Second
	}
	if maxShardSize > 0 {
		config.MaxShardSize = maxShardSize
	}
	config.PreserveWAL = preserveWAL
	if throttle > 0 {
		config.Throttle = throttle
	}
	if fragmentationThreshold >= 0 && fragmentationThreshold <= 1.0 {
		config.FragmentationThreshold = fragmentationThreshold
	}
	if minFreeSpace > 0 {
		config.MinFreeSpace = minFreeSpace
	}
	if schedule != "" {
		config.Schedule = schedule
	}

	// Validate required parameters
	if config.RootPath == "" {
		log.Fatal("--root is required")
	}

	// Create and start background vacuumer
	bv := NewBackgroundVacuumer(config)
	if err := bv.Start(); err != nil {
		log.Fatalf("Failed to start background vacuumer: %v", err)
	}

	// Wait for interrupt signal
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c

	log.Println("Received interrupt signal")
	bv.Stop()
}
