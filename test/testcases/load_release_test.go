//go:build L0

package testcases

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test load collection
func TestLoadCollection(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// load two replicas
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// check replicas
	replicas, errReplicas := mc.GetReplicas(ctx, collName)
	common.CheckErr(t, errReplicas, true)
	require.Len(t, replicas, 1)

	// check collection loaded
	collection, _ := mc.DescribeCollection(ctx, collName)
	require.True(t, collection.Loaded)
}

// test load not existed collection
func TestLoadCollectionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// Load collection
	errLoad := mc.LoadCollection(ctx, "collName", false)
	common.CheckErr(t, errLoad, false, "exist")
}

// test load collection async
func TestLoadCollectionAsync(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// load collection async
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, true)

	for {
		time.Sleep(2 * time.Second)

		// describe collection
		collection, errDescribe := mc.DescribeCollection(ctx, collName)
		if errDescribe == nil {
			if collection.Loaded {
				break
			}
		} else {
			t.FailNow()
		}
	}
}

// load collection without index
func TestLoadCollectionWithoutIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, false)
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, false, "index not found")

	// load partitions without index
	errLoadPartition := mc.LoadPartitions(ctx, collName, []string{common.DefaultPartition}, true)
	common.CheckErr(t, errLoadPartition, false, "index not found")
}

// load collection with multi partitions
func TestLoadCollectionMultiPartitions(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// check all partitions loaded
	partitions, _ := mc.ShowPartitions(ctx, collName)
	for _, partition := range partitions {
		require.True(t, partition.Loaded)
	}

	// check collection loaded
	collection, _ := mc.DescribeCollection(ctx, collName)
	require.True(t, collection.Loaded)
}

// test load with empty partition name ""
func TestLoadEmptyPartitionName(t *testing.T) {
	//t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/373")
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition with empty partition names
	errLoadEmpty := mc.LoadPartitions(ctx, collName, []string{""}, false)
	common.CheckErr(t, errLoadEmpty, false, "request failed")
}

// test load partitions with empty slice []string{}
func TestLoadEmptyPartitionSlice(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition with empty partition names
	errLoadEmpty := mc.LoadPartitions(ctx, collName, []string{}, false)
	common.CheckErr(t, errLoadEmpty, false, "due to no partition specified")
}

// test load partitions not exist
func TestLoadPartitionsNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load with not exist partition names
	errLoadNotExist := mc.LoadPartitions(ctx, collName, []string{"xxx"}, false)
	common.CheckErr(t, errLoadNotExist, false, fmt.Sprintf("partition xxx of collection %s does not exist", collName))

	// load partition with part exist partition names
	errLoadPartExist := mc.LoadPartitions(ctx, collName, []string{"xxx", partitionName}, false)
	common.CheckErr(t, errLoadPartExist, false, fmt.Sprintf("partition xxx of collection %s does not exist", collName))
}

// test load partition
func TestLoadPartitions(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/375")
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	nb := 1000
	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, nb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName}, false)
	common.CheckErr(t, errLoad, true)

	// check collection loaded false
	collection, _ := mc.DescribeCollection(ctx, collName)
	require.False(t, collection.Loaded)

	partitions, _ := mc.ShowPartitions(ctx, collName)
	for _, p := range partitions {
		if p.Name == partitionName {
			require.True(t, p.Loaded)
		} else {
			require.True(t, p.Loaded)
		}
		log.Printf("id: %d, name: %s, loaded %t", p.ID, p.Name, p.Loaded)
	}

	//query from nb from partition
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, int64(nb)})
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition, []entity.Column{
		entity.NewColumnInt64(common.DefaultIntFieldName, []int64{int64(nb)}),
	})
}

// test load multi partition
func TestLoadMultiPartitions(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/375")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName, common.DefaultPartition}, false)
	common.CheckErr(t, errLoad, true)

	//query from nb from partition
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, common.DefaultNb})
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition, []entity.Column{
		entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb}),
	})
}

// test load partitions repeatedly
func TestLoadPartitionsRepeatedly(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/375")
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load repeated partition names
	errLoadRepeat := mc.LoadPartitions(ctx, collName, []string{partitionName, partitionName}, false)
	common.CheckErr(t, errLoadRepeat, true)

	//query from nb from partition
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb})
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition, []entity.Column{queryIds})
}

// test load partitions async
func TestLoadPartitionsAsync(t *testing.T) {
	t.Skipf("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partitions async
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName}, true)
	common.CheckErr(t, errLoad, true)

	// check load results
	for {
		time.Sleep(time.Second * 5)

		// check partition loaded
		partitions, errShow := mc.ShowPartitions(ctx, collName)
		if errShow == nil {
			for _, p := range partitions {
				log.Printf("id: %d, name: %s, loaded %t", p.ID, p.Name, p.Loaded)
				if p.Name == partitionName && p.Loaded {
					break
				}
			}
		} else {
			t.FailNow()
		}
	}
}

func TestLoadCollectionMultiVectors(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// create hnsw index on part vector field and load -> load failed
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	for _, field := range []string{common.DefaultFloatVecFieldName, common.DefaultBinaryVecFieldName} {
		err := mc.CreateIndex(ctx, collName, field, indexHnsw, false)
		common.CheckErr(t, err, true)
	}

	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, false, "there is no vector index on field")

	// create index for another vector field
	indexScann, _ := entity.NewIndexSCANN(entity.COSINE, 16, false)
	for _, field := range []string{common.DefaultFloat16VecFieldName, common.DefaultBFloat16VecFieldName} {
		err := mc.CreateIndex(ctx, collName, field, indexScann, false)
		common.CheckErr(t, err, true)
	}
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
}

// test release partition
func TestReleasePartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection -> insert data -> create index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	//load collection
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, true)

	// release collection
	errRelease := mc.ReleaseCollection(ctx, collName)
	common.CheckErr(t, errRelease, true)

	// check collection loaded
	collection, _ := mc.DescribeCollection(ctx, collName)
	require.False(t, collection.Loaded)

	// check release success
	_, errQuery := mc.QueryByPks(ctx, collName, []string{}, entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0}),
		[]string{common.DefaultIntFieldName})
	// TODO change error msg or code
	common.CheckErr(t, errQuery, false, "not loaded")
}

// test release not exist collection
func TestReleaseCollectionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection -> insert data -> create index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	//load collection
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, true)

	// release collection
	errRelease := mc.ReleaseCollection(ctx, "collName")
	common.CheckErr(t, errRelease, false, "not exist")
}

// test release partitions
func TestReleasePartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partitions async
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName}, true)
	common.CheckErr(t, errLoad, true)

	// release partition
	errRelease := mc.ReleasePartitions(ctx, collName, []string{partitionName})
	common.CheckErr(t, errRelease, true)

	// check release success
	partitions, _ := mc.ShowPartitions(ctx, collName)
	for _, p := range partitions {
		if p.Name == partitionName {
			require.False(t, p.Loaded)
		}
	}

	// check release success
	_, errQuery := mc.QueryByPks(ctx, collName, []string{partitionName}, entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb}),
		[]string{common.DefaultIntFieldName})

	// TODO fix error msg or code
	common.CheckErr(t, errQuery, false, "not loaded")
}

// test release partition not exist -> error or part exist -> success
func TestReleasePartitionsNotExist(t *testing.T) {
	t.Skipf("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/375")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partitions
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName}, false)
	common.CheckErr(t, errLoad, true)

	// release partition
	errRelease := mc.ReleasePartitions(ctx, collName, []string{"partitionName"})
	common.CheckErr(t, errRelease, false, "not exist")

	// release partition
	errRelease2 := mc.ReleasePartitions(ctx, collName, []string{"partitionName", partitionName})
	common.CheckErr(t, errRelease2, false, "not exist")

	// check release success
	partitions, _ := mc.ShowPartitions(ctx, collName)
	for _, p := range partitions {
		if p.Name == partitionName {
			require.False(t, p.Loaded)
		}
	}

	// check release success
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb})
	_, errQuery := mc.QueryByPks(ctx, collName, []string{partitionName}, queryIds,
		[]string{common.DefaultIntFieldName})
	common.CheckErr(t, errQuery, false, "not loaded into memory when query")
}

func TestReleaseMultiPartitions(t *testing.T) {
	t.Skipf("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/375")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load multi partitions
	errLoad := mc.LoadPartitions(ctx, collName, []string{partitionName, common.DefaultPartition}, false)
	common.CheckErr(t, errLoad, true)

	// release partition
	errRelease := mc.ReleasePartitions(ctx, collName, []string{partitionName, common.DefaultPartition})
	common.CheckErr(t, errRelease, true)

	// check release success
	partitions, _ := mc.ShowPartitions(ctx, collName)
	for _, p := range partitions {
		require.False(t, p.Loaded)
	}

	// check release success
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, common.DefaultNb})
	_, errQuery := mc.QueryByPks(ctx, collName, []string{partitionName, common.DefaultPartition}, queryIds,
		[]string{common.DefaultIntFieldName})
	common.CheckErr(t, errQuery, false, "not loaded into memory when query")
}
