package client

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMetaCache(t *testing.T) {
	meta := &metaCache{
		sessionTsMap: make(map[string]uint64),
		collInfoMap:  make(map[string]collInfo),
	}

	t.Run("session-ts-get", func(t *testing.T) {
		ts, ok := meta.getSessionTs("")
		assert.False(t, ok)
		assert.Equal(t, uint64(0), ts)
	})

	t.Run("session-ts-set-then-get", func(t *testing.T) {
		meta.setSessionTs("0", 1)
		ts, ok := meta.getSessionTs("0")
		assert.True(t, ok)
		assert.Equal(t, uint64(1), ts)
	})

	t.Run("session-ts-monotonic-set", func(t *testing.T) {
		meta.setSessionTs("0", 2)
		meta.setSessionTs("0", 1)
		ts, ok := meta.getSessionTs("0")
		assert.True(t, ok)
		assert.Equal(t, uint64(2), ts)
	})

	t.Run("info-get", func(t *testing.T) {
		info, ok := meta.getCollectionInfo("")
		assert.False(t, ok)
		assert.Nil(t, info)
	})

	t.Run("info-set-get", func(t *testing.T) {
		info1 := &collInfo{
			Name: "aaa",
		}
		meta.setCollectionInfo(info1.Name, info1)
		info2, ok := meta.getCollectionInfo(info1.Name)
		assert.Equal(t, info1, info2)
		assert.True(t, ok)
		meta.setCollectionInfo(info1.Name, nil)
		info2, ok = meta.getCollectionInfo(info1.Name)
		assert.Nil(t, info2)
		assert.False(t, ok)
	})
}
