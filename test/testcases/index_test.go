//go:build L0

package testcases

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	entity "github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test create index with supported float vector index, L2 metric type
func TestCreateIndex(t *testing.T) {
	t.Parallel()

	// create index
	allFloatIndexes := common.GenAllFloatIndex(entity.L2)
	for _, idx := range allFloatIndexes {
		ctx := createContext(t, time.Second*common.DefaultTimeout*3)
		// connect
		mc := createMilvusClient(ctx, t)
		// create default collection with flush data
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName("my_index"))
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex("my_index", idx.IndexType(), idx.Params())
		common.CheckIndexResult(t, indexes, expIndex)
	}
}

func TestCreateIndexString(t *testing.T) {
	t.Skipf("Not supported create index on varchar field, issue: %s", "https://github.com/milvus-io/milvus-sdk-go/issues/362")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createVarcharCollectionWithDataIndex(ctx, t, mc, true)
	err := mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, nil, false, client.WithIndexName("my_index"))
	common.CheckErr(t, err, true)
}

// test create index with supported float vector index, Ip metric type
func TestCreateIndexIp(t *testing.T) {
	t.Parallel()

	// create index
	allFloatIndexes := common.GenAllFloatIndex(entity.IP)
	for _, idx := range allFloatIndexes {
		ctx := createContext(t, time.Second*common.DefaultTimeout*3)
		// connect
		mc := createMilvusClient(ctx, t)
		// create default collection with flush data
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName("my_index"))
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex("my_index", idx.IndexType(), idx.Params())
		common.CheckIndexResult(t, indexes, expIndex)
	}
}

// test create index with supported binary vector index
func TestCreateIndexBinary(t *testing.T) {
	t.Skipf("Issue: %s", "https://github.com/milvus-io/milvus-sdk-go/issues/351")
	t.Parallel()

	// create index
	metricTypes := []entity.MetricType{
		entity.JACCARD,
		entity.HAMMING,
		entity.SUBSTRUCTURE,
		entity.SUPERSTRUCTURE,
	}
	for _, metricType := range metricTypes {
		allFloatIndexes := common.GenAllBinaryIndex(metricType)
		for _, idx := range allFloatIndexes {
			ctx := createContext(t, time.Second*common.DefaultTimeout)
			// connect
			mc := createMilvusClient(ctx, t)
			// create default collection with flush data
			collName, _ := createBinaryCollectionWithDataIndex(ctx, t, mc, false, false)
			err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName("my_index"))
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultBinaryVecFieldName)
			expIndex := entity.NewGenericIndex("my_index", idx.IndexType(), idx.Params())
			common.CheckIndexResult(t, indexes, expIndex)
		}
	}
}

// test create index without specify index name
func TestCreateIndexWithoutName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe index return index with default name
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, idx.IndexType(), idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)
}

// test new index by Generic index
func TestCreateIndexGeneric(t *testing.T) {
	t.Skipf("Issue: %s", "https://github.com/milvus-io/milvus-sdk-go/issues/351")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	IvfFlatParams := map[string]string{"nlist": "128"}
	idx := entity.NewGenericIndex("my_index", entity.IvfFlat, IvfFlatParams)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, idx.IndexType(), idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)
}

// test create index with not exist index name
func TestCreateIndexNotExistCollName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, "haha", common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, false, "collection haha does not exist")
}

func TestCreateIndexNotExistField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, "exist", idx, false)
	common.CheckErr(t, err, false, "does not exist")
}

// test create index on non-vector field
func TestCreateIndexNotSupportedField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, idx, false)
	common.CheckErr(t, err, false, "is not vector field")
}

// test create index with invalid params
// https://github.com/milvus-io/milvus-sdk-go/issues/357
func TestCreateIndexInvalidParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// invalid IvfFlat nlist [1, 65536]
	_, errIvfFlatNlist := entity.NewIndexIvfFlat(entity.L2, 0)
	_, errIvfFlatNlist2 := entity.NewIndexIvfFlat(entity.L2, 65537)
	common.CheckErr(t, errIvfFlatNlist, false, "nlist not valid")
	common.CheckErr(t, errIvfFlatNlist2, false, "nlist not valid")

	// invalid IvfSq8 nlist [1, 65536]
	_, errIvfSq8Nlist := entity.NewIndexIvfFlat(entity.L2, 0)
	_, errIvfSq8Nlist2 := entity.NewIndexIvfFlat(entity.L2, 65537)
	common.CheckErr(t, errIvfSq8Nlist, false, "nlist not valid")
	common.CheckErr(t, errIvfSq8Nlist2, false, "nlist not valid")

	// invalid IvfPq nlist [1, 65536]
	_, errIvfPqNlist := entity.NewIndexIvfPQ(entity.L2, -1, 16, 8)
	common.CheckErr(t, errIvfPqNlist, false, "nlist not valid")
	_, errIvfPqNlist2 := entity.NewIndexIvfPQ(entity.L2, 65538, 16, 8)
	common.CheckErr(t, errIvfPqNlist2, false, "nlist not valid")

	// invalid IvfPq params m dim ≡ 0 (mod m), nbits [1, 16]
	_, errIvfPqNbits := entity.NewIndexIvfPQ(entity.L2, 128, 8, 0)
	common.CheckErr(t, errIvfPqNbits, false, "nbits not valid")
	_, errIvfPqNbits2 := entity.NewIndexIvfPQ(entity.L2, 128, 8, 17)
	common.CheckErr(t, errIvfPqNbits2, false, "nbits not valid")
	// TODO unclear error message
	idxInvalidm, _ := entity.NewIndexIvfPQ(entity.L2, 128, 7, 8)
	errm := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxInvalidm, false)
	common.CheckErr(t, errm, false, "dimension must")

	// invalid Hnsw M [4, 64], efConstruction [8, 512]
	_, errHnswM := entity.NewIndexHNSW(entity.L2, 3, 96)
	common.CheckErr(t, errHnswM, false, "M not valid")
	_, errHnswM2 := entity.NewIndexHNSW(entity.L2, 128, 96)
	common.CheckErr(t, errHnswM2, false, "M not valid")
	_, errHnswEf := entity.NewIndexHNSW(entity.L2, 8, 7)
	common.CheckErr(t, errHnswEf, false, "efConstruction not valid")
	_, errHnswEf2 := entity.NewIndexHNSW(entity.L2, 8, 515)
	common.CheckErr(t, errHnswEf2, false, "efConstruction not valid")

	// invalid Annoy n_trees [1, 1024]
	_, errAnnoyNTrees := entity.NewIndexANNOY(entity.L2, 0)
	common.CheckErr(t, errAnnoyNTrees, false, "n_trees not valid")
	_, errAnnoyNTrees2 := entity.NewIndexANNOY(entity.L2, 2048)
	common.CheckErr(t, errAnnoyNTrees2, false, "n_trees not valid")

	// invalid flat metric type jaccard
	// TODO unclear error message
	idx, _ := entity.NewIndexFlat(entity.JACCARD)
	errMetricType := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, errMetricType, false, "metric type not found or not supported")
}

// test create index with nil index
func TestCreateIndexNil(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/358")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, nil, false)
	common.CheckErr(t, err, false, "invalid index")
}

// test create index async true
func TestCreateIndexAsync(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/361")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, true)

	for {
		time.Sleep(time.Second * 10)
		indexState, errState := mc.GetIndexState(ctx, collName, common.DefaultFloatVecFieldName)
		if errState == nil {
			if indexState == 3 {
				break
			}
		} else {
			t.FailNow()
		}
	}
}

// test get index state
func TestIndexState(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/361")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// get index state
	state, errState := mc.GetIndexState(ctx, collName, common.DefaultFloatVecFieldName)
	common.CheckErr(t, errState, true)
	require.Equal(t, entity.IndexState(common.IndexStateValue["Finished"]), state, "Expected finished index state")
}

func TestIndexStateNotExistCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// get index state
	_, errState := mc.GetIndexState(ctx, "collName", common.DefaultFloatVecFieldName)
	common.CheckErr(t, errState, false, "does not exist")
}

func TestDropIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// drop index
	errDrop := mc.DropIndex(ctx, collName, common.DefaultFloatVecFieldName)
	common.CheckErr(t, errDrop, true)
	indexes, errDescribe := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	common.CheckErr(t, errDescribe, false, "index doesn't exist, collectionID")
	require.Nil(t, indexes)
}
