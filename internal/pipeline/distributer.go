package pipeline

// Distributer maps the global pool of shards to a single reader. It generalizes
// the per-worker round-robin sharding to an outer DDP rank dimension (feature
// F2: distributed training).
//
// Without distributed training (WorldSize <= 1) there is exactly one rank, so
// the global reader index collapses to WorkerID over NumWorkers total readers —
// byte-for-byte the original `shardIndex % NumWorkers == WorkerID` rule, which
// keeps every existing single-process path unchanged.
//
// Under DDP, torchrun launches WorldSize processes (ranks). Each rank runs its
// own DataLoader with NumWorkers workers, so across the whole job there are
// WorldSize*NumWorkers independent readers. Reader (rank, workerID) takes global
// index rank*NumWorkers + workerID, and a shard at sorted position i is served
// by the reader where i % TotalReaders == GlobalReaderID. This is the single
// guarantee DDP needs from the data layer: every reader's shard set is disjoint
// and their union is the whole dataset exactly once, so no sample is fed to two
// ranks in an epoch (which would bias the averaged gradient).
//
// Deployment assumption: one daemon per rank. On a multi-node job that is one
// daemon per node; on a single host the ranks' daemons must use distinct ports
// + SHM files + pipe paths (the Python client parameterizes all three), since
// the sharding math here only partitions *which shards* a reader serves, not the
// physical SHM/pipe transport.
type Distributer struct {
	rank       int
	worldSize  int
	workerID   int
	numWorkers int
}

// NewDistributer builds the shard mapper for one reader from its WorkerConfig.
// A zero/absent WorldSize is normalized to 1 (non-distributed), so callers that
// never set the distributed fields get the legacy behavior.
func NewDistributer(cfg WorkerConfig) Distributer {
	ws := cfg.WorldSize
	if ws < 1 {
		ws = 1
	}
	return Distributer{
		rank:       cfg.Rank,
		worldSize:  ws,
		workerID:   cfg.WorkerID,
		numWorkers: cfg.NumWorkers,
	}
}

// TotalReaders is the number of independent readers across the whole job
// (all ranks × all workers per rank).
func (d Distributer) TotalReaders() int { return d.worldSize * d.numWorkers }

// GlobalReaderID is this reader's index in [0, TotalReaders).
func (d Distributer) GlobalReaderID() int { return d.rank*d.numWorkers + d.workerID }

// Owns reports whether the shard at sorted position i belongs to this reader.
func (d Distributer) Owns(i int) bool {
	return i%d.TotalReaders() == d.GlobalReaderID()
}
