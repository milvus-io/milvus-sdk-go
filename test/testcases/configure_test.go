//go:build L3

package testcases

import (
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// config queryNode.replicas >=2 -> test load with multi replicas
func TestLoadCollectionReplicas(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// load two replicas
	errLoad := mc.LoadCollection(ctx, collName, false, client.WithReplicaNumber(2))
	common.CheckErr(t, errLoad, true)

	// check replicas and segment info
	replicas, errReplicas := mc.GetReplicas(ctx, collName)
	require.Len(t, replicas, 2)
	common.CheckErr(t, errReplicas, true)
	for _, replica := range replicas {
		require.Len(t, replica.ShardReplicas, int(common.DefaultShards))
		for _, shard := range replica.ShardReplicas {
			log.Printf("ReplicaID: %v, NodeIDs: %v, LeaderID: %v, NodesIDs: %v, DmChannelName: %v",
				replica.ReplicaID, replica.NodeIDs, shard.LeaderID, shard.NodesIDs, shard.DmChannelName)
		}
	}

	// check segment info
	segments, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	common.CheckPersistentSegments(t, segments, int64(common.DefaultNb))
	for _, segment := range segments {
		log.Printf("segmentId: %d, NumRows: %d, CollectionID: %d, ParititionID %d, IndexId: %d, State %v",
			segment.ID, segment.NumRows, segment.CollectionID, segment.ParititionID, segment.IndexID, segment.State)
	}
}

// config common.retentionDuration=40 -> test compact after delete half
func TestCompactAfterDelete(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/379")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with 1 shard
	collName := createDefaultCollection(ctx, t, mc, true, 1)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// delete half ids
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:common.DefaultNb/2])
	errDelete := mc.DeleteByPks(ctx, collName, "", deleteIds)
	common.CheckErr(t, errDelete, true)

	// compact
	time.Sleep(common.RetentionDuration + 1)
	compactionID, _ := mc.Compact(ctx, collName, 0)
	mc.WaitForCompactionCompleted(ctx, compactionID)

	// get compaction plans
	_, plans, _ := mc.GetCompactionStateWithPlans(ctx, compactionID)
	log.Println(plans)
}
