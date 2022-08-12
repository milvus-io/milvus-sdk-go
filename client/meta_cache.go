package client

import (
	"sync"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// Magical timestamps for communicating with server
const (
	StrongTimestamp     uint64 = 0
	EventuallyTimestamp uint64 = 1
	BoundedTimestamp    uint64 = 2
)

type collInfo struct {
	ID               int64          // collection id
	Name             string         // collection name
	Schema           *entity.Schema // collection schema, with fields schema and primary key definition
	ConsistencyLevel entity.ConsistencyLevel
}

var MetaCache = metaCache{
	sessionTsMap: make(map[string]uint64),
	collInfoMap:  make(map[string]collInfo),
}

// timestampMap collects the last-write-timestamp of every collection, which is required by session consistency level.
type metaCache struct {
	sessionMu    sync.RWMutex
	colMu        sync.RWMutex
	sessionTsMap map[string]uint64 // collectionName -> last-write-timestamp
	collInfoMap  map[string]collInfo
}

func (m *metaCache) getSessionTs(cName string) (uint64, bool) {
	m.sessionMu.RLock()
	defer m.sessionMu.RUnlock()
	ts, ok := m.sessionTsMap[cName]
	return ts, ok
}

func (m *metaCache) setSessionTs(cName string, ts uint64) {
	m.sessionMu.Lock()
	defer m.sessionMu.Unlock()
	m.sessionTsMap[cName] = max(m.sessionTsMap[cName], ts) // increase monotonically
}

func (m *metaCache) setCollectionInfo(cName string, c *collInfo) {
	m.colMu.Lock()
	defer m.colMu.Unlock()
	if c == nil {
		delete(m.collInfoMap, cName)
	} else {
		m.collInfoMap[cName] = *c
	}
}

func (m *metaCache) getCollectionInfo(cName string) (*collInfo, bool) {
	m.colMu.RLock()
	defer m.colMu.RUnlock()
	col, ok := m.collInfoMap[cName]
	if !ok {
		return nil, false
	}
	return &collInfo{
		ID:               col.ID,
		Name:             col.Name,
		Schema:           col.Schema,
		ConsistencyLevel: col.ConsistencyLevel,
	}, true
}

func max(x, y uint64) uint64 {
	if x > y {
		return x
	}
	return y
}
