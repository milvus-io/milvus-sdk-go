package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScalarIndex(t *testing.T) {
	oldScalarIdx := NewScalarIndex()

	assert.EqualValues(t, "", oldScalarIdx.IndexType(), "use AUTO index when index type not provided")
	_, has := oldScalarIdx.Params()[tIndexType]
	assert.False(t, has)

	idxWithType := NewScalarIndexWithType(Sorted)

	assert.EqualValues(t, Sorted, idxWithType.IndexType())
	assert.EqualValues(t, Sorted, idxWithType.Params()[tIndexType])

	idxWithType = NewScalarIndexWithType(Trie)

	assert.EqualValues(t, Trie, idxWithType.IndexType())
	assert.EqualValues(t, Trie, idxWithType.Params()[tIndexType])
}
