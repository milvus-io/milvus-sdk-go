//go:build L0

package testcases

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func supportScalarIndexFieldType(field entity.FieldType) bool {
	vectorFieldTypes := []entity.FieldType{
		entity.FieldTypeBinaryVector, entity.FieldTypeFloatVector,
		entity.FieldTypeFloat16Vector, entity.FieldTypeBFloat16Vector,
		entity.FieldTypeArray, entity.FieldTypeJSON,
	}
	for _, vectorFieldType := range vectorFieldTypes {
		if field == vectorFieldType {
			return false
		}
	}
	return true
}

// test create index with supported float vector index, L2 metric type
func TestCreateIndex(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	allFloatIndexes := common.GenAllFloatIndex()
	for _, idx := range allFloatIndexes {
		// create default collection with flush data
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(common.DefaultIndexName))
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex(common.DefaultIndexName, idx.IndexType(), idx.Params())
		common.CheckIndexResult(t, indexes, expIndex)

		// drop index
		err = mc.DropIndex(ctx, collName, common.DefaultFloatVecFieldName)
		common.CheckErr(t, err, true)
	}
}

// create index for fp16 and bf16 vectors
func TestCreateIndexMultiVectors(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true, ShardsNum: 1,
		Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors, start: 0,
		nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true}
	_, _ = insertData(ctx, t, mc, dp)
	_ = mc.Flush(ctx, collName, false)

	// create index for all vectors
	idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false)
	for _, idx := range common.GenAllFloatIndex() {
		for _, fieldName := range []string{common.DefaultFloat16VecFieldName, common.DefaultBFloat16VecFieldName} {
			err := mc.CreateIndex(ctx, collName, fieldName, idx, false, client.WithIndexName(fieldName))
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
			expIndex := entity.NewGenericIndex(fieldName, idx.IndexType(), idx.Params())
			common.CheckIndexResult(t, indexes, expIndex)

			// drop index
			err = mc.DropIndex(ctx, collName, fieldName, client.WithIndexName(fieldName))
			common.CheckErr(t, err, true)
		}
	}
	for _, metricType := range common.SupportBinIvfFlatMetricType {
		idx, _ := entity.NewIndexBinFlat(metricType, 64)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName(common.DefaultBinaryVecFieldName))
		common.CheckErr(t, err, true)

		// drop index
		err = mc.DropIndex(ctx, collName, common.DefaultBinaryVecFieldName, client.WithIndexName(common.DefaultBinaryVecFieldName))
		common.CheckErr(t, err, true)
	}
}

// test create index on same field twice
func TestCreateIndexDup(t *testing.T) {
	// create index
	idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	idxIvfSq8, _ := entity.NewIndexIvfSQ8(entity.L2, 128)
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)
	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultFloatVecFieldName, idxHnsw.IndexType(), idxHnsw.Params())
	common.CheckIndexResult(t, indexes, expIndex)

	err = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxIvfSq8, false)
	common.CheckErr(t, err, false, "CreateIndex failed: at most one distinct index is allowed per field")
}

// test create scalar index on all scalar field
func TestCreateScalarIndex(t *testing.T) {
	t.Skip("https://github.com/milvus-io/milvus-sdk-go/issues/671")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	//connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// create index for all vector fields
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	for _, field := range common.AllVectorsFieldsName {
		err := mc.CreateIndex(ctx, collName, field, indexHnsw, false)
		common.CheckErr(t, err, true)
		indexes, _ := mc.DescribeIndex(ctx, collName, field)
		expIndex := entity.NewGenericIndex(field, entity.HNSW, indexHnsw.Params())
		common.CheckIndexResult(t, indexes, expIndex)
	}

	coll, _ := mc.DescribeCollection(ctx, collName)
	idx := entity.NewScalarIndex()
	for _, field := range coll.Schema.Fields {
		if supportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
			expIndex := entity.NewGenericIndex(field.Name, "", idx.Params())
			common.CheckIndexResult(t, indexes, expIndex)
		}
	}
}

// test create scalar index with vector field name
func TestCreateIndexWithOtherFieldName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	//connect
	mc := createMilvusClient(ctx, t)

	collName, _ := createVarcharCollectionWithDataIndex(ctx, t, mc, false)
	idx := entity.NewScalarIndex()
	// create index with vector field name as index name (vector field name is the vector default index name)
	err := mc.CreateIndex(ctx, collName, common.DefaultVarcharFieldName, idx, false,
		client.WithIndexName(common.DefaultBinaryVecFieldName))
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultVarcharFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultBinaryVecFieldName, "", idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)

	// create index in binary field with default name
	err = mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false)
	common.CheckErr(t, err, false, "CreateIndex failed: at most one distinct index is allowed per field")
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
	common.CheckErr(t, err, false, "create index on JSON field is not supported")

	// create scalar index on json field
	//err = mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, entity.NewScalarIndex(), false, client.WithIndexName("json_index"))
	//common.CheckErr(t, err, false, "create index on json field is not supported")
}

func TestCreateIndexArrayField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create scalar and vector index on array field
	scalarIdx := entity.NewScalarIndex()
	vectorIdx, _ := entity.NewIndexSCANN(entity.L2, 10, false)
	collection, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection.Schema.Fields {
		if field.DataType == entity.FieldTypeArray {
			// create scalar index
			err := mc.CreateIndex(ctx, collName, field.Name, scalarIdx, false, client.WithIndexName("scalar_index"))
			common.CheckErr(t, err, false, "create index on Array field is not supported: invalid parameter")
			// create vector index
			err1 := mc.CreateIndex(ctx, collName, field.Name, vectorIdx, false, client.WithIndexName("vector_index"))
			common.CheckErr(t, err1, false, "create index on Array field is not supported: invalid parameter")
		}
	}
}

// test create index with supported binary vector index
func TestCreateIndexBinaryFlat(t *testing.T) {
	t.Parallel()

	// create index
	for _, metricType := range common.SupportBinFlatMetricType {
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
	for _, metricType := range common.SupportBinIvfFlatMetricType {
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

	// create BinIvfFlat, BinFlat index with not supported metric type
	invalidMetricTypes := []entity.MetricType{
		entity.L2,
		entity.COSINE,
		entity.IP,
		entity.TANIMOTO,
	}
	for _, metricType := range invalidMetricTypes {
		// create BinFlat
		idxBinFlat, _ := entity.NewIndexBinFlat(metricType, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinFlat, false, client.WithIndexName("my_index"))
		common.CheckErr(t, err, false, "supported: [HAMMING JACCARD SUBSTRUCTURE SUPERSTRUCTURE]")
	}

	invalidMetricTypes2 := []entity.MetricType{
		entity.L2,
		entity.COSINE,
		entity.IP,
		entity.TANIMOTO,
		entity.SUBSTRUCTURE,
		entity.SUPERSTRUCTURE,
	}

	for _, metricType := range invalidMetricTypes2 {
		// create BinIvfFlat index
		idxBinIvfFlat, _ := entity.NewIndexBinIvfFlat(metricType, 128)
		errIvf := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinIvfFlat, false, client.WithIndexName("my_index2"))
		common.CheckErr(t, errIvf, false, fmt.Sprintf("metric type %v not found or not supported", metricType))
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
	expIndex := entity.NewGenericIndex(common.DefaultFloatVecFieldName, idx.IndexType(), idx.Params())
	common.CheckIndexResult(t, indexes, expIndex)
}

// test create auto index
func TestCreateIndexWithoutIndexTypeParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

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
	expIndex := entity.NewGenericIndex(common.DefaultFloatVecFieldName, entity.AUTOINDEX, expParams)
	common.CheckIndexResult(t, indexes, expIndex)
}

// test new index by Generic index
func TestCreateIndexGeneric(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	IvfFlatParams := map[string]string{"nlist": "128", "metric_type": "L2"}
	idx := entity.NewGenericIndex("my_index", entity.IvfFlat, IvfFlatParams)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultFloatVecFieldName, idx.IndexType(), idx.Params())
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
	common.CheckErr(t, err, false, "index type not match")

	// create scann index
	indexScann, _ := entity.NewIndexSCANN(entity.L2, 8, true)
	err = mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, indexScann, false)
	common.CheckErr(t, err, false, "index type not match")
}

// test create index with invalid params
func TestCreateIndexInvalidParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// invalid IvfFlat nlist [1, 65536]
	_, errIvfFlatNlist := entity.NewIndexIvfFlat(entity.L2, 0)
	_, errIvfFlatNlist2 := entity.NewIndexIvfFlat(entity.L2, 65537)
	common.CheckErr(t, errIvfFlatNlist, false, "nlist has to be in range [1, 65536]")
	common.CheckErr(t, errIvfFlatNlist2, false, "nlist has to be in range [1, 65536]")

	// invalid IvfSq8 nlist [1, 65536]
	_, errIvfSq8Nlist := entity.NewIndexIvfFlat(entity.L2, 0)
	_, errIvfSq8Nlist2 := entity.NewIndexIvfFlat(entity.L2, 65537)
	common.CheckErr(t, errIvfSq8Nlist, false, "nlist has to be in range [1, 65536]")
	common.CheckErr(t, errIvfSq8Nlist2, false, "nlist has to be in range [1, 65536]")

	// invalid IvfPq nlist [1, 65536]
	_, errIvfPqNlist := entity.NewIndexIvfPQ(entity.L2, -1, 16, 8)
	common.CheckErr(t, errIvfPqNlist, false, "nlist has to be in range [1, 65536]")
	_, errIvfPqNlist2 := entity.NewIndexIvfPQ(entity.L2, 65538, 16, 8)
	common.CheckErr(t, errIvfPqNlist2, false, "nlist has to be in range [1, 65536]")

	// invalid IvfPq params m dim ≡ 0 (mod m), nbits [1, 16]
	_, errIvfPqNbits := entity.NewIndexIvfPQ(entity.L2, 128, 8, 0)
	common.CheckErr(t, errIvfPqNbits, false, "nbits has to be in range [1, 16]")
	_, errIvfPqNbits2 := entity.NewIndexIvfPQ(entity.L2, 128, 8, 17)
	common.CheckErr(t, errIvfPqNbits2, false, "nbits has to be in range [1, 16]")
	// TODO unclear error message
	idxInvalidm, _ := entity.NewIndexIvfPQ(entity.L2, 128, 7, 8)
	errm := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxInvalidm, false)
	// TODO change error message check
	common.CheckErr(t, errm, false, "dimension")

	// invalid Hnsw M [4, 64], efConstruction [8, 512]
	_, errHnswM := entity.NewIndexHNSW(entity.L2, 3, 96)
	common.CheckErr(t, errHnswM, false, "M has to be in range [4, 64]")
	_, errHnswM2 := entity.NewIndexHNSW(entity.L2, 128, 96)
	common.CheckErr(t, errHnswM2, false, "M has to be in range [4, 64]")
	_, errHnswEf := entity.NewIndexHNSW(entity.L2, 8, 7)
	common.CheckErr(t, errHnswEf, false, "efConstruction has to be in range [8, 512]")
	_, errHnswEf2 := entity.NewIndexHNSW(entity.L2, 8, 515)
	common.CheckErr(t, errHnswEf2, false, "efConstruction has to be in range [8, 512]")

	// invalid Scann nlist [1, 65536], with_raw_data [true, false], no default
	for _, nlist := range []int{0, 65536 + 1} {
		_, errScann := entity.NewIndexSCANN(entity.L2, nlist, true)
		log.Println(errScann)
		common.CheckErr(t, errScann, false, "nlist has to be in range [1, 65536]")
	}
	for _, mt := range []entity.MetricType{entity.HAMMING, entity.JACCARD, entity.SUBSTRUCTURE, entity.SUPERSTRUCTURE} {
		idxScann, errScann2 := entity.NewIndexSCANN(mt, 64, true)
		common.CheckErr(t, errScann2, true)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxScann, false)
		common.CheckErr(t, err, false,
			fmt.Sprintf("metric type %s not found or not supported, supported: [L2 IP COSINE]", mt))
	}

	// invalid flat metric type jaccard for flat index
	idx, _ := entity.NewIndexFlat(entity.JACCARD)
	errMetricType := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, errMetricType, false,
		"metric type JACCARD not found or not supported, supported: [L2 IP COSINE]")
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

// create same index name on different vector field
func TestIndexMultiVectorDupName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, err := insertData(ctx, t, mc, dp)
	common.CheckErr(t, err, true)

	// create index with same indexName on different fields
	idx, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	err = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName("index_1"))
	common.CheckErr(t, err, true)

	// same index on another field
	err = mc.CreateIndex(ctx, collName, common.DefaultFloat16VecFieldName, idx, false, client.WithIndexName("index_1"))
	common.CheckErr(t, err, false, "reateIndex failed: at most one distinct index is allowed per field")
}

// test get index state
func TestIndexState(t *testing.T) {
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
	common.CheckErr(t, errDescribe, false, "index not found")
	require.Nil(t, indexes)
}

func TestDropIndexCreateIndex(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/385")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with flush data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// describe collection
	collection, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection.Schema.Fields {
		log.Printf("field name: %v, field TypeParams: %v, field IndexParams: %v)", field.Name, field.TypeParams, field.IndexParams)
	}

	// drop index
	errDrop := mc.DropIndex(ctx, collName, common.DefaultFloatVecFieldName)
	common.CheckErr(t, errDrop, true)
	indexes, errDescribe := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	common.CheckErr(t, errDescribe, false, "index doesn't exist, collectionID")
	require.Nil(t, indexes)

	// create IP index
	ipIdx, _ := entity.NewIndexHNSW(entity.IP, 8, 96)
	err2 := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, ipIdx, false)
	common.CheckErr(t, err2, true)

	// describe index
	ipIndexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	log.Println(ipIndexes[0].Params())

	// describe collection
	collection2, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection2.Schema.Fields {
		log.Printf("field name: %v, field TypeParams: %v, field IndexParams: %v)", field.Name, field.TypeParams, field.IndexParams)
	}
}
