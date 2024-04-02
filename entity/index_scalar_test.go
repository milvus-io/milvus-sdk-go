package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScalarIndex(t *testing.T) {
	oldScalarIdx := NewScalarIndex()

	assert.EqualValues(t, "", oldScalarIdx.IndexType(), "use AUTO index when index type not provided")

	idxWithType := NewScalarIndexWithType(Sorted)

	assert.EqualValues(t, Sorted, idxWithType.IndexType())
}
