//go:build L0

package testcases

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestCompact(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with 1 shard
	cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: 1, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	for i := 0; i < 4; i++ {
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: i * common.DefaultNb, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)
	}

	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, field := range common.AllFloatVectorsFieldNames {
		err := mc.CreateIndex(ctx, collName, field, indexHnsw, false)
		common.CheckErr(t, err, true)
	}
	err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, indexBinary, false)
	common.CheckErr(t, err, true)

	// get persisted segments
	segments, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	require.Len(t, segments, 4)
	segIds := make([]int64, 0, 4)
	for _, seg := range segments {
		require.Equal(t, seg.NumRows, int64(common.DefaultNb))
		segIds = append(segIds, seg.ID)
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
	require.Equal(t, compactionState2, entity.CompactionStateCompleted)
	var targetSeg int64
	for _, plan := range compactionPlan {
		if plan.PlanType == entity.CompactionPlanMergeSegments {
			require.ElementsMatch(t, segIds, plan.Source)
			targetSeg = plan.Target
		}
	}

	// get persisted segments
	segments2, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	var actualSeg []int64
	actualRows := 0
	for _, s := range segments2 {
		actualSeg = append(actualSeg, s.ID)
		actualRows = int(s.NumRows) + actualRows
	}
	require.Equal(t, common.DefaultNb*4, actualRows)
	require.ElementsMatch(t, []int64{targetSeg}, actualSeg)
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
