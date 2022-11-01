package entity

import (
	"testing"

	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	"github.com/stretchr/testify/assert"
)

func TestSegmentFlushed(t *testing.T) {
	segment := &Segment{}
	assert.False(t, segment.Flushed())
	segment.State = common.SegmentState_Growing
	assert.False(t, segment.Flushed())
	segment.State = common.SegmentState_Flushing
	assert.False(t, segment.Flushed())
	segment.State = common.SegmentState_Flushed
	assert.True(t, segment.Flushed())
}
