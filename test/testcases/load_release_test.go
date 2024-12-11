//go:build L0

package testcases

import (
	"fmt"
	"log"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test load collection
func TestLoadCollection(t *testing.T) {
	// t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
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
	log.Println(collection.Loaded)
	// require.True(t, collection.Loaded)
}

// test load not existed collection
func TestLoadCollectionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// Load collection
	errLoad := mc.LoadCollection(ctx, "collName", false)
	common.CheckErr(t, errLoad, false, "collection not found")
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
	// t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/374")
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
		log.Println(partition.Loaded)
		// require.True(t, partition.Loaded)
	}

	// check collection loaded
	collection, _ := mc.DescribeCollection(ctx, collName)
	log.Println(collection.Loaded)
	// require.True(t, collection.Loaded)
}

// test load with empty partition name "" -> default partition
func TestLoadEmptyPartitionName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	nb := 500
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	parName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, nb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load partition with empty partition names -> actually default
	errLoadEmpty := mc.LoadPartitions(ctx, collName, []string{""}, false)
	common.CheckErr(t, errLoadEmpty, true)

	countDef, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(nb), countDef.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	_, errPar := mc.Query(ctx, collName, []string{parName}, "", []string{common.QueryCountFieldName})
	common.CheckErr(t, errPar, false, "partition not loaded")
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
	common.CheckErr(t, errLoadNotExist, false, "partition not found")

	// load partition with part exist partition names
	errLoadPartExist := mc.LoadPartitions(ctx, collName, []string{"xxx", partitionName}, false)
	common.CheckErr(t, errLoadPartExist, false, "partition not found")
}

// test load partition
func TestLoadPartitions(t *testing.T) {
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
			// require.True(t, p.Loaded)
			log.Println(p.Loaded)
		} else {
			log.Println(p.Loaded)
			// require.True(t, p.Loaded)
		}
		log.Printf("id: %d, name: %s, loaded %t", p.ID, p.Name, p.Loaded)
	}

	// query from nb from partition
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, int64(nb)})
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition, []entity.Column{
		entity.NewColumnInt64(common.DefaultIntFieldName, []int64{int64(nb)}),
	})
}

// test load multi partition
func TestLoadMultiPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithConsistencyLevel(entity.ClStrong))
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load default partition
	errLoad := mc.LoadPartitions(ctx, collName, []string{common.DefaultPartition}, false)
	common.CheckErr(t, errLoad, true)

	// query nb from default partition
	resDef, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.EqualValues(t, common.DefaultNb, resDef.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// load partition and query -> actually not loaded
	errLoad = mc.LoadPartitions(ctx, collName, []string{partitionName}, false)
	common.CheckErr(t, errLoad, true)
	resPar, _ := mc.Query(ctx, collName, []string{partitionName}, "", []string{common.QueryCountFieldName})
	require.EqualValues(t, common.DefaultNb, resPar.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	res, _ := mc.Query(ctx, collName, []string{}, "", []string{common.QueryCountFieldName})
	require.EqualValues(t, common.DefaultNb*2, res.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
}

// test load partitions repeatedly
func TestLoadPartitionsRepeatedly(t *testing.T) {
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

	// query from nb from partition
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb})
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition, []entity.Column{queryIds})
}

// test load partitions async
func TestLoadPartitionsAsync(t *testing.T) {
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
		state, err := mc.GetLoadState(ctx, collName, []string{partitionName})
		// partitions, errShow := mc.ShowPartitions(ctx, collName)
		if err == nil {
			if state == entity.LoadStateLoaded {
				return
			}
		} else {
			t.FailNow()
		}
	}
}

func TestLoadCollectionMultiVectors(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// create hnsw index on part vector field and load -> load failed
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	for _, field := range []string{common.DefaultFloatVecFieldName, common.DefaultFloat16VecFieldName, common.DefaultBFloat16VecFieldName} {
		err := mc.CreateIndex(ctx, collName, field, indexHnsw, false)
		common.CheckErr(t, err, true)
	}

	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, false, "there is no vector index on field")

	// create index for another vector field
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, field := range []string{common.DefaultBinaryVecFieldName} {
		err := mc.CreateIndex(ctx, collName, field, indexBinary, false)
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

	// load collection
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

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, true)

	// release collection
	errRelease := mc.ReleaseCollection(ctx, "collName")
	common.CheckErr(t, errRelease, false, "collection not found")
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
	common.CheckErr(t, errRelease, false, "partition not found")

	// release partition
	errRelease2 := mc.ReleasePartitions(ctx, collName, []string{"partitionName", partitionName})
	common.CheckErr(t, errRelease2, false, "partition not found")

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
	common.CheckErr(t, errQuery, true)
}

func TestReleaseMultiPartitions(t *testing.T) {
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
	common.CheckErr(t, errQuery, false, "collection not loaded")
}

// test mmap collection raw data and index
// create -> insert -> flush -> index with mmap -> load -> alter collection with mmap -> reload -> read op
func TestMmapCollectionIndexDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	dp := DataParams{
		DoInsert: true, CollectionName: collName, CollectionFieldsType: AllFields, start: 0,
		nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true,
	}
	insertData(ctx, t, mc, dp)
	_ = mc.Flush(ctx, collName, false)

	// create vector index with mmap
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			err := mc.CreateIndex(ctx, collName, fieldName, indexBinary, false, client.WithMmap(true))
			common.CheckErr(t, err, true)
		} else if fieldName == common.DefaultFloatVecFieldName {
			err := mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false, client.WithMmap(true))
			common.CheckErr(t, err, true)
		} else {
			err := mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
			common.CheckErr(t, err, true)
		}
	}

	// describe index and check mmap
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultFloatVecFieldName || fieldName == common.DefaultBinaryVecFieldName {
			indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
			require.Len(t, indexes, 1)
			// check index mmap
			require.Equal(t, "true", indexes[0].Params()["mmap.enabled"])
		}
	}

	// load collection -> describe collection and check mmap false
	_ = mc.LoadCollection(ctx, collName, false)
	coll, _ := mc.DescribeCollection(ctx, collName)
	require.Equal(t, "", coll.Properties["mmap.enabled"])

	// alter collection and check collection mmap
	_ = mc.ReleaseCollection(ctx, collName)
	err := mc.AlterCollection(ctx, collName, entity.Mmap(true))
	common.CheckErr(t, err, true)

	// describe collection
	mc.LoadCollection(ctx, collName, false)
	coll, _ = mc.DescribeCollection(ctx, collName)
	require.Equal(t, "true", coll.Properties["mmap.enabled"])

	// query
	queryRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < 10", common.DefaultIntFieldName), []string{"*"})
	common.CheckErr(t, err, true)
	require.Equal(t, queryRes.GetColumn(common.DefaultIntFieldName).Len(), 10)
	common.CheckOutputFields(t, queryRes, common.GetAllFieldsName(true, false))

	// search
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))

	// hybrid search
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(3)),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
}

// test mmap collection raw data and index
// create -> insert -> flush -> index -> load -> alter collection with mmap -> alter index with mmap -> reload -> read op
func TestMmapAlterCollectionIndexDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0,
		nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true,
	}

	ips := GenDefaultIndexParamsForAllVectors()
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// describe index and check mmap
	for _, fieldName := range common.AllVectorsFieldsName {
		indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
		// check index mmap
		require.Equal(t, "", indexes[0].Params()["mmap.enabled"])
	}

	// describe collection
	mc.LoadCollection(ctx, collName, false)
	coll, _ := mc.DescribeCollection(ctx, collName)
	require.Equal(t, "", coll.Properties["mmap.enabled"])

	// alter mmap: collection and index
	_ = mc.ReleaseCollection(ctx, collName)
	err := mc.AlterCollection(ctx, collName, entity.Mmap(true))
	common.CheckErr(t, err, true)
	for _, fieldName := range common.AllVectorsFieldsName {
		err := mc.AlterIndex(ctx, collName, fieldName, client.WithMmap(true))
		common.CheckErr(t, err, true)
	}

	// load collection -> describe collection and check mmap false
	// describe collection
	mc.LoadCollection(ctx, collName, false)
	coll, _ = mc.DescribeCollection(ctx, collName)
	require.Equal(t, "true", coll.Properties["mmap.enabled"])
	for _, fieldName := range common.AllVectorsFieldsName {
		idx, _ := mc.DescribeIndex(ctx, collName, fieldName)
		require.Equal(t, "true", idx[0].Params()["mmap.enabled"])
	}

	// query
	queryRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < 10", common.DefaultIntFieldName), []string{"*"})
	common.CheckErr(t, err, true)
	require.Equal(t, queryRes.GetColumn(common.DefaultIntFieldName).Len(), 10)
	common.CheckOutputFields(t, queryRes, common.GetAllFieldsName(true, false))

	// search
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))

	// hybrid search
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(3)),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
}

// test mmap collection loaded
func TestMmapCollectionLoaded(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: Int64FloatVec, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// mmap collection raw data
	err := mc.AlterCollection(ctx, collName, entity.Mmap(true))
	common.CheckErr(t, err, false, "can not alter mmap properties if collection loaded")

	// mmap index
	err = mc.AlterIndex(ctx, collName, common.DefaultFloatVecFieldName, client.WithMmap(true))
	common.CheckErr(t, err, false, "can't alter index on loaded collection, please release the collection first")
}

// test mmap collection which scalar field indexed
func TestMmapCollectionScalarIndexed(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: Int64FloatVec, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create scalar index
	for _, fName := range []string{common.DefaultIntFieldName, common.DefaultFloatFieldName} {
		err := mc.CreateIndex(ctx, collName, fName, entity.NewScalarIndex(), false)
		common.CheckErr(t, err, true)
	}

	// mmap collection
	mc.ReleaseCollection(ctx, collName)
	err := mc.AlterCollection(ctx, collName, entity.Mmap(true))
	common.CheckErr(t, err, true)
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	sp, _ := entity.NewIndexHNSWSearchParam(32)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, resSearch, common.DefaultNq, common.DefaultTopK)
	common.CheckOutputFields(t, resSearch[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName})
}

// test mmap scalar index: inverted
func TestMmapScalarInvertedIndex(t *testing.T) {
	// vector index
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: true,
	}

	// build vector's indexes
	ips := GenDefaultIndexParamsForAllVectors()
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create scalar index with mmap
	collection, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, entity.NewScalarIndexWithType(entity.Inverted), false, client.WithMmap(true))
			common.CheckErr(t, err, true)
		}
	}

	// load and search
	mc.LoadCollection(ctx, collName, false)
	// query
	queryRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < 10", common.DefaultIntFieldName), []string{"*"})
	common.CheckErr(t, err, true)
	require.Equal(t, queryRes.GetColumn(common.DefaultIntFieldName).Len(), 10)
	common.CheckOutputFields(t, queryRes, common.GetAllFieldsName(true, false))

	// search
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))

	// hybrid search
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(3)),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
}

// test mmap scalar index: bitmap
func TestMmapScalarBitmapIndex(t *testing.T) {
	// vector index
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: true,
	}

	// build vector's indexes
	ips := GenDefaultIndexParamsForAllVectors()
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create scalar index with mmap
	collection, _ := mc.DescribeCollection(ctx, collName)
	BitmapNotSupport := []interface{}{entity.FieldTypeJSON, entity.FieldTypeDouble, entity.FieldTypeFloat}
	for _, field := range collection.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) && !field.PrimaryKey && !(common.CheckContainsValue(BitmapNotSupport, field.DataType) || (field.DataType == entity.FieldTypeArray && common.CheckContainsValue(BitmapNotSupport, field.ElementType))) {
			err := mc.CreateIndex(ctx, collName, field.Name, entity.NewScalarIndexWithType(entity.Bitmap), false, client.WithMmap(true))
			common.CheckErr(t, err, true)
		}
	}

	// load and search
	mc.LoadCollection(ctx, collName, false)
	// query
	queryRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < 10", common.DefaultIntFieldName), []string{"*"})
	common.CheckErr(t, err, true)
	require.Equal(t, queryRes.GetColumn(common.DefaultIntFieldName).Len(), 10)
	common.CheckOutputFields(t, queryRes, common.GetAllFieldsName(true, false))

	// search
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))

	// hybrid search
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(3)),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
}

// test mmap scalar index: bitmap
func TestMmapScalarHybirdIndex(t *testing.T) {
	// vector index
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: true,
	}

	// build vector's indexes
	ips := GenDefaultIndexParamsForAllVectors()
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create scalar index with mmap
	collection, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, entity.NewScalarIndex(), false, client.WithMmap(true))
			common.CheckErr(t, err, true)
		}
	}

	// load and search
	mc.LoadCollection(ctx, collName, false)
	// query
	queryRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < 10", common.DefaultIntFieldName), []string{"*"})
	common.CheckErr(t, err, true)
	require.Equal(t, queryRes.GetColumn(common.DefaultIntFieldName).Len(), 10)
	common.CheckOutputFields(t, queryRes, common.GetAllFieldsName(true, false))

	// search
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))

	// hybrid search
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(3)),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
}

// test mmap unsupported index: DiskANN, GPU-class, scalar index except inverted
func TestMmapIndexUnsupported(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// create index with mmap
	idx, _ := entity.NewIndexDISKANN(entity.COSINE)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithMmap(true))
	common.CheckErr(t, err, false, "index type DISKANN does not support mmap")

	err1 := mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, entity.NewScalarIndexWithType(entity.Trie), false, client.WithMmap(true))
	common.CheckErr(t, err1, false, "index type Trie does not support mmap")

	err1 = mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, entity.NewScalarIndexWithType(entity.Sorted), false, client.WithMmap(true))
	common.CheckErr(t, err1, false, "index type STL_SORT does not support mmap")
}

// test mmap unsupported index: DiskANN, GPU-class
func TestMmapScalarAutoIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	dp := DataParams{
		DoInsert: true, CollectionName: collName, CollectionFieldsType: AllFields, start: 0,
		nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: false,
	}
	insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// mmap not supported HYBRID index
	err1 := mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, entity.NewScalarIndex(), false, client.WithMmap(true))
	common.CheckErr(t, err1, true)

	// mmap not supported HYBRID index
	err1 = mc.CreateIndex(ctx, collName, common.DefaultBoolFieldName, entity.NewScalarIndexWithType(entity.Bitmap), false, client.WithMmap(true))
	common.CheckErr(t, err1, true)
}

func TestAlterIndexMmapUnsupportedIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// diskAnn
	idxDiskAnn, _ := entity.NewIndexDISKANN(entity.IP)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxDiskAnn, false)
	common.CheckErr(t, err, true)
	err = mc.AlterIndex(ctx, collName, common.DefaultFloatVecFieldName, client.WithMmap(true))
	common.CheckErr(t, err, false, "index type DISKANN does not support mmap")

	// bitmap index with mmap, create bitmap index on primary key not supported
	err = mc.CreateIndex(ctx, collName, common.DefaultIntFieldName, entity.NewScalarIndexWithType(entity.Bitmap), false)
	common.CheckErr(t, err, false, "create bitmap index on primary key not supported")

	// HYBRID index
	err = mc.CreateIndex(ctx, collName, common.DefaultInt32FieldName, entity.NewScalarIndex(), false)
	common.CheckErr(t, err, true)
	errAlert := mc.AlterIndex(ctx, collName, common.DefaultInt32FieldName, client.WithMmap(true))
	common.CheckErr(t, errAlert, true)

	// Trie index
	err = mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, entity.NewScalarIndexWithType(entity.Trie), false)
	common.CheckErr(t, err, true)
	errAlert = mc.AlterIndex(ctx, collName, common.DefaultVarcharFieldName, client.WithMmap(true))
	common.CheckErr(t, errAlert, false, "index type Trie does not support mmap")

	// STL_SORT
	err = mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, entity.NewScalarIndexWithType(entity.Sorted), false)
	common.CheckErr(t, err, true)
	errAlert = mc.AlterIndex(ctx, collName, common.DefaultFloatFieldName, client.WithMmap(true))
	common.CheckErr(t, errAlert, false, "index type STL_SORT does not support mmap")
}

func TestMmapAlterIndex(t *testing.T) {
	t.Parallel()
	for _, mmap := range []bool{true, false} {
		ctx := createContext(t, time.Second*common.DefaultTimeout*2)
		// connect
		mc := createMilvusClient(ctx, t)

		// create -> insert -> flush -> index -> load
		cp := CollectionParams{
			CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp)

		dp := DataParams{
			DoInsert: true, CollectionName: collName, CollectionFieldsType: Int64FloatVec, start: 0,
			nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: false,
		}
		insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		// create index and enable mmap
		idxHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false, client.WithMmap(!mmap))
		common.CheckErr(t, err, true)

		// alter index and enable mmap
		err = mc.AlterIndex(ctx, collName, common.DefaultFloatVecFieldName, client.WithMmap(mmap))
		common.CheckErr(t, err, true)

		idx, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		require.Equal(t, strconv.FormatBool(mmap), idx[0].Params()["mmap.enabled"])

		err = mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, err, true)

		queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
		searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec1, common.DefaultFloatVecFieldName,
			entity.COSINE, common.DefaultTopK, sp)
		common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName})
	}
}

// test search when mmap sparse collection
func TestMmapSparseCollection(t *testing.T) {
	t.Skip("sparse index support mmap now")
	t.Parallel()
	idxInverted, _ := entity.NewIndexSparseInverted(entity.IP, 0)
	idxWand, _ := entity.NewIndexSparseWAND(entity.IP, 0)
	for _, idx := range []entity.Index{idxInverted, idxWand} {
		ctx := createContext(t, time.Second*common.DefaultTimeout*2)
		// connect
		mc := createMilvusClient(ctx, t)

		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen,
		}

		dp := DataParams{
			DoInsert: true, CollectionFieldsType: Int64VarcharSparseVec, start: 0, nb: common.DefaultNb * 5,
			dim: common.DefaultDim, EnableDynamicField: true,
		}

		// index params
		idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		ips := []IndexParams{
			{BuildIndex: true, Index: idx, FieldName: common.DefaultSparseVecFieldName, async: false},
			{BuildIndex: true, Index: idxHnsw, FieldName: common.DefaultFloatVecFieldName, async: false},
		}
		collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

		// alter mmap
		mc.ReleaseCollection(ctx, collName)
		// alter index and enable mmap
		err := mc.AlterIndex(ctx, collName, common.DefaultSparseVecFieldName, client.WithMmap(true))
		common.CheckErr(t, err, false, fmt.Sprintf("index type %s does not support mmap", idx.IndexType()))
		err = mc.AlterIndex(ctx, collName, common.DefaultFloatVecFieldName, client.WithMmap(true))
		common.CheckErr(t, err, true)
		err = mc.AlterCollection(ctx, collName, entity.Mmap(true))
		common.CheckErr(t, err, true)
		err = mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, err, true)

		// search with floatVec field
		outputFields := []string{
			common.DefaultIntFieldName, common.DefaultVarcharFieldName, common.DefaultFloatVecFieldName,
			common.DefaultSparseVecFieldName, common.DefaultDynamicFieldName,
		}
		queryVecFloat := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexSparseInvertedSearchParam(0)
		resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVecFloat, common.DefaultFloatVecFieldName,
			entity.L2, common.DefaultTopK, sp)
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
		common.CheckOutputFields(t, resSearch[0].Fields, outputFields)

		// search with sparse vector field
		queryVecSparse := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeSparseVector)
		resSearch, errSearch = mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVecSparse, common.DefaultSparseVecFieldName,
			entity.IP, common.DefaultTopK, sp)
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
		common.CheckOutputFields(t, resSearch[0].Fields, outputFields)
	}
}
