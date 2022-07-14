package client

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestTimestampMap(t *testing.T) {
	tsm := &timestampMap{m: map[int64]uint64{}}

	t.Run("get", func(t *testing.T) {
		ts, ok := tsm.get(0)
		assert.False(t, ok)
		assert.Equal(t, uint64(0), ts)
	})

	t.Run("set-then-get", func(t *testing.T) {
		tsm.set(0, 1)
		ts, ok := tsm.get(0)
		assert.True(t, ok)
		assert.Equal(t, uint64(1), ts)
	})

	t.Run("monotonic-set", func(t *testing.T) {
		tsm.set(0, 2)
		tsm.set(0, 1)
		ts, ok := tsm.get(0)
		assert.True(t, ok)
		assert.Equal(t, uint64(2), ts)
	})
}
