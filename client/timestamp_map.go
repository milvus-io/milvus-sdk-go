package client

import (
	"sync"
)

// Magical timestamps for communicating with server
const (
	StrongTimestamp     uint64 = 0
	EventuallyTimestamp uint64 = 1
	BoundedTimestamp    uint64 = 2
)

// global timestampMap
var tsm = timestampMap{
	m: make(map[int64]uint64),
}

// timestampMap collects the last-write-timestamp of every collection, which is required by session consistency level.
type timestampMap struct {
	mu sync.RWMutex
	m  map[int64]uint64 // collectionID -> last-write-timestamp
}

func (tsm *timestampMap) get(cid int64) (uint64, bool) {
	tsm.mu.RLock()
	defer tsm.mu.RUnlock()
	ts, ok := tsm.m[cid]
	return ts, ok
}

func (tsm *timestampMap) set(cid int64, ts uint64) {
	tsm.mu.Lock()
	defer tsm.mu.Unlock()
	tsm.m[cid] = max(tsm.m[cid], ts) // increase monotonically
}

func max(x, y uint64) uint64 {
	if x > y {
		return x
	}
	return y
}
