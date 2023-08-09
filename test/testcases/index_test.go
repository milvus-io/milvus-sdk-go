//go:build L0

package testcases

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

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
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName("my_index"))
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex("my_index", idx.IndexType(), idx.Params())
		common.CheckIndexResult(t, indexes, expIndex)
	}
}

// test create index for varchar field
func TestCreateIndexString(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createVarcharCollectionWithDataIndex(ctx, t, mc, false)
	idx := entity.NewScalarIndex()
	err := mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, idx, false, client.WithIndexName("scalar_index"))
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultVarcharFieldName)
	expIndex := entity.NewGenericIndex("scalar_index", "", idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)
}

func TestCreateIndexJsonField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecJSON,
		AutoID:               false,
		EnableDynamicField:   false,
		ShardsNum:            common.DefaultShards,
		Dim:                  common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName:       collName,
		PartitionName:        "",
		CollectionFieldsType: Int64FloatVecJSON,
		start:                0,
		nb:                   common.DefaultNb,
		dim:                  common.DefaultDim,
		EnableDynamicField:   false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	// create vector index on json field
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, idx, false, client.WithIndexName("json_index"))
	common.CheckErr(t, err, false, "create index on json field is not supported")

	// create scalar index on json field
	err = mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, entity.NewScalarIndex(), false, client.WithIndexName("json_index"))
	common.CheckErr(t, err, false, "create index on json field is not supported")
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
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName("my_index"))
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex("my_index", idx.IndexType(), idx.Params())
		common.CheckIndexResult(t, indexes, expIndex)
	}
}

// test create index with supported binary vector index
func TestCreateIndexBinaryFlat(t *testing.T) {
	t.Parallel()

	// create index
	metricTypes := []entity.MetricType{
		entity.JACCARD,
		entity.HAMMING,
		entity.TANIMOTO,
		entity.SUBSTRUCTURE,
		entity.SUPERSTRUCTURE,
	}
	for _, metricType := range metricTypes {
		idx, _ := entity.NewIndexBinFlat(metricType, 128)
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

// test create index with supported binary vector index
func TestCreateIndexBinaryIvfFlat(t *testing.T) {
	t.Parallel()

	// create index
	metricTypes := []entity.MetricType{
		entity.JACCARD,
		entity.HAMMING,
		entity.TANIMOTO,
	}
	for _, metricType := range metricTypes {
		idx, _ := entity.NewIndexBinIvfFlat(metricType, 128)
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

// test create binary index with unsupported metrics type
func TestCreateBinaryIndexNotSupportedMetricsType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createBinaryCollectionWithDataIndex(ctx, t, mc, false, false)

	// create BinFlat index with metric type L2
	idxBinFlat, _ := entity.NewIndexBinFlat(entity.L2, 128)
	err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinFlat, false, client.WithIndexName("my_index"))
	common.CheckErr(t, err, false, "supported: [HAMMING JACCARD TANIMOTO SUBSTRUCTURE SUPERSTRUCTURE]")

	// create BinIvfFlat index with invalid metric type
	invalidMetricTypes := []entity.MetricType{
		entity.SUBSTRUCTURE,
		entity.SUPERSTRUCTURE,
		entity.L2,
	}
	for _, metricType := range invalidMetricTypes {
		// create BinIvfFlat index
		idxBinIvfFlat, _ := entity.NewIndexBinIvfFlat(metricType, 128)
		errIvf := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinIvfFlat, false, client.WithIndexName("my_index2"))
		common.CheckErr(t, errIvf, false, "supported: [HAMMING JACCARD TANIMOTO]")
	}

}

// test create index without specify index name
func TestCreateIndexWithoutName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe index return index with default name
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, idx.IndexType(), idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)
}

// test create auto index
func TestCreateIndexWithoutIndexTypeParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

	// create index
	idx := entity.NewGenericIndex("", "", nil)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expParams := map[string]string{
		"metric_type": string(entity.IP),
		"index_type":  string(entity.AUTOINDEX),
	}
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.AUTOINDEX, expParams)
	common.CheckIndexResult(t, indexes, expIndex)
}

// test new index by Generic index
func TestCreateIndexGeneric(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

	// create index
	IvfFlatParams := map[string]string{"nlist": "128", "metric_type": "L2"}
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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, idx, false)
	common.CheckErr(t, err, false, "can only use 'STL_SORT' when is scalar field except VarChar")
}

// test create index with invalid params
// https://github.com/milvus-io/milvus-sdk-go/issues/357
func TestCreateIndexInvalidParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)
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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false, false)

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
