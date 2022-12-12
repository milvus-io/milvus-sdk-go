//go:build L3

package testcases

import (
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

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
