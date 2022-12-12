//go:build L0

package testcases

import (
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestCompact(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/379")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with 1 shard
	collName := createDefaultCollection(ctx, t, mc, true, 1)

	// insert data
	for i := 0; i < 2; i++ {
		_, floatColumn, vecColumn := common.GenDefaultColumnData((i+1)*common.DefaultNb, common.DefaultNb, common.DefaultDim)
		_, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		mc.Flush(ctx, collName, false)
	}

	// get persisted segments
	segments, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	for _, s := range segments {
		log.Printf("Id: %d, numRows: %d", s.ID, s.NumRows)
	}

	// compact
	compactionID, errCompact := mc.Compact(ctx, collName, 100)
	common.CheckErr(t, errCompact, true)

	// get compaction states
	errWait := mc.WaitForCompactionCompleted(ctx, compactionID)
	common.CheckErr(t, errWait, true)

	// get compaction plan
	compactionState2, compactionPlan, errPlan := mc.GetCompactionStateWithPlans(ctx, compactionID)
	common.CheckErr(t, errPlan, true)
	log.Println(compactionState2)
	log.Println(compactionPlan)

	// get persisted segments
	segments2, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	for _, s := range segments2 {
		log.Printf("Id: %d, numRows: %d", s.ID, s.NumRows)
	}
}

// test compaction collection not exist
func TestCompactCollectionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	_, err := mc.Compact(ctx, "coll", 0)
	common.CheckErr(t, err, false, "collection coll does not exist")
}

// test compact empty collection
func TestCompactEmptyCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with 1 shard
	collName := createDefaultCollection(ctx, t, mc, true, 1)

	// compact
	compactionID, err := mc.Compact(ctx, collName, 0)
	common.CheckErr(t, err, true)

	mc.WaitForCompactionCompleted(ctx, compactionID)
	compactionState, compactionPlans, errPlans := mc.GetCompactionStateWithPlans(ctx, compactionID)
	common.CheckErr(t, errPlans, true)
	require.Equal(t, compactionState, entity.CompactionStateCompleted)
	require.Len(t, compactionPlans, 0)
}
