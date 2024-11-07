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

/*
Info about scalar index
TRIE: only support varchar
STL_SORT: only support numeric (not include Array field)
INVERTED: all supported except Json
Bitmap: all supported except Json, float, double. (If Array field, according to its ElementType)
ScalarAutoIndex: {"int_*": "HYBRID","varchar": "HYBRID","bool": "BITMAP", "float/double": "INVERTED"}
	- except Json
	- if Array field, according to its ElementType
*/

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
	cp := CollectionParams{
		CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true, ShardsNum: 1,
		Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors, start: 0,
		nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true,
	}
	_, _ = insertData(ctx, t, mc, dp)
	_ = mc.Flush(ctx, collName, false)

	// create index for all vectors
	idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false)
	common.CheckErr(t, err, true)
	for _, idx := range common.GenAllFloatIndex() {
		for _, fieldName := range []string{common.DefaultFloat16VecFieldName, common.DefaultBFloat16VecFieldName} {
			log.Printf("index name=%s, index type=%v, index params=%v", idx.Name(), idx.IndexType(), idx.Params())
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

func TestDescribeIndexMultiVectors(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)
	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb * 2,
		dim: common.DefaultDim, EnableDynamicField: true,
	}

	// create index for all vector fields
	ips := GenDefaultIndexParamsForAllVectors()

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))
	for _, fieldName := range common.AllVectorsFieldsName {
		indexes, err := mc.DescribeIndex(ctx, collName, fieldName)
		common.CheckErr(t, err, true)
		require.Len(t, indexes, 1)
		for _, ip := range ips {
			if ip.FieldName == fieldName {
				expIndex := entity.NewGenericIndex(fieldName, ip.Index.IndexType(), ip.Index.Params())
				common.CheckIndexResult(t, indexes, expIndex)
			}
		}
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
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// create index for all vector fields
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	var expIndex entity.Index
	var err error
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			err = mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
			expIndex = entity.NewGenericIndex(fieldName, entity.BinIvfFlat, indexBinary.Params())
		} else {
			err = mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
			expIndex = entity.NewGenericIndex(fieldName, entity.HNSW, indexHnsw.Params())
		}
		common.CheckErr(t, err, true)
		indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
		common.CheckIndexResult(t, indexes, expIndex)
	}

	coll, _ := mc.DescribeCollection(ctx, collName)
	common.PrintAllFieldNames(collName, coll.Schema)
	idx := entity.NewScalarIndex()
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
			expIndex := entity.NewGenericIndex(field.Name, "", idx.Params())
			common.CheckIndexResult(t, indexes, expIndex)
		}
	}
	// load -> search and output all fields
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchRes, _ := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))
}

// test create scalar index on loaded collection
func TestCreateIndexOnLoadedCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// create index for all vector fields
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	var expIndex entity.Index
	var err error
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			err = mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
			expIndex = entity.NewGenericIndex(fieldName, entity.BinIvfFlat, indexBinary.Params())
		} else {
			err = mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
			expIndex = entity.NewGenericIndex(fieldName, entity.HNSW, indexHnsw.Params())
		}
		common.CheckErr(t, err, true)
		indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
		common.CheckIndexResult(t, indexes, expIndex)
	}

	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	coll, _ := mc.DescribeCollection(ctx, collName)
	idx := entity.NewScalarIndex()
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
			// need check failed
			common.CheckErr(t, err, true, "")
		}
	}

	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			_, err := mc.DescribeIndex(ctx, collName, field.Name)
			common.CheckErr(t, err, true, "")
		}
	}
}

// Trie scalar index only supported on varchar
func TestCreateTrieScalarIndexUnsupportedDataType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	// index params
	ips := []IndexParams{{BuildIndex: false}}
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create Trie scalar index on varchar field
	coll, _ := mc.DescribeCollection(ctx, collName)
	idx := entity.NewScalarIndexWithType(entity.Trie)
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			if field.DataType == entity.FieldTypeVarChar {
				err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, true)

				// describe index
				indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
				expIndex := entity.NewGenericIndex(field.Name, entity.Trie, idx.Params())
				common.CheckIndexResult(t, indexes, expIndex)
			} else {
				err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, false, "TRIE are only supported on varchar field")
			}
		}
	}
}

// Sort scalar index only supported on numeric field
func TestCreateSortScalarIndexUnsupportedDataType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	// index params
	ips := []IndexParams{{BuildIndex: false}}
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create Trie scalar index on varchar field
	coll, _ := mc.DescribeCollection(ctx, collName)
	idx := entity.NewScalarIndexWithType(entity.Sorted)
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			if field.DataType == entity.FieldTypeVarChar || field.DataType == entity.FieldTypeBool ||
				field.DataType == entity.FieldTypeArray {
				err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, false, "STL_SORT are only supported on numeric field")
			} else {
				log.Println(field.Name, field.DataType)
				err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, true)
				// describe index
				indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
				expIndex := entity.NewGenericIndex(field.Name, entity.Sorted, idx.Params())
				common.CheckIndexResult(t, indexes, expIndex)
			}
		}
	}
}

// create Inverted index for all scalar fields
func TestCreateInvertedScalarIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	// index params
	ips := GenDefaultIndexParamsForAllVectors()
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create Trie scalar index on varchar field
	coll, _ := mc.DescribeCollection(ctx, collName)
	idx := entity.NewScalarIndexWithType(entity.Inverted)
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
			require.Len(t, indexes, 1)
			log.Println(indexes[0].Name(), indexes[0].IndexType(), indexes[0].Params())
			expIndex := entity.NewGenericIndex(field.Name, entity.Inverted, idx.Params())
			common.CheckIndexResult(t, indexes, expIndex)
		}
	}
	// load -> search and output all fields
	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(false, false))
}

// create Bitmap index for all scalar fields
func TestCreateBitmapScalarIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	// build vector's indexes
	ips := GenDefaultIndexParamsForAllVectors()
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// create BITMAP scalar index
	coll, _ := mc.DescribeCollection(ctx, collName)
	common.PrintAllFieldNames(collName, coll.Schema)
	idx := entity.NewScalarIndexWithType(entity.Bitmap)
	BitmapNotSupport := []interface{}{entity.FieldTypeJSON, entity.FieldTypeDouble, entity.FieldTypeFloat}
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			log.Println(field.Name, field.DataType, field.ElementType)
			if common.CheckContainsValue(BitmapNotSupport, field.DataType) || (field.DataType == entity.FieldTypeArray && common.CheckContainsValue(BitmapNotSupport, field.ElementType)) {
				err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, false, "bitmap index are only supported")
			} else {
				if field.PrimaryKey {
					err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
					common.CheckErr(t, err, false, "create bitmap index on primary key not supported")
				} else {
					err := mc.CreateIndex(ctx, collName, field.Name, idx, false, client.WithIndexName(field.Name))
					common.CheckErr(t, err, true)

					// describe index
					indexes, _ := mc.DescribeIndex(ctx, collName, field.Name)
					require.Len(t, indexes, 1)
					log.Println(indexes[0].Name(), indexes[0].IndexType(), indexes[0].Params())
					expIndex := entity.NewGenericIndex(field.Name, entity.Bitmap, idx.Params())
					common.CheckIndexResult(t, indexes, expIndex)
				}
			}
		}
	}
	// load -> search and output all fields
	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10 && %s >= 0 && %s <= 3000 || %s > 10 || %s == true", common.DefaultIntFieldName, common.DefaultInt8FieldName, common.DefaultInt16FieldName, common.DefaultInt32FieldName, common.DefaultBoolFieldName)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(false, false))
}

// test create index on vector field
func TestCreateScalarIndexVectorField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{
		CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}

	dp := DataParams{
		DoInsert: true, CollectionFieldsType: AllVectors, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}

	// no index
	ips := []IndexParams{{BuildIndex: false}}
	lp := LoadParams{DoLoad: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithLoadParams(lp),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	for _, ip := range []entity.IndexType{entity.Sorted, entity.Trie, entity.Inverted, entity.Bitmap} {
		idx := entity.NewScalarIndexWithType(ip)
		for _, fieldName := range common.AllVectorsFieldsName {
			err := mc.CreateIndex(ctx, collName, fieldName, idx, false)
			common.CheckErr(t, err, false, "metric type not set for vector index")
		}
	}
	for _, fieldName := range common.AllFloatVectorsFieldNames {
		idxDefault := entity.NewScalarIndex()
		err := mc.CreateIndex(ctx, collName, fieldName, idxDefault, false)
		common.CheckErr(t, err, true)
		descIndex, _ := mc.DescribeIndex(ctx, collName, fieldName)
		require.Equal(t, entity.AUTOINDEX, descIndex[0].IndexType())
	}
}

// test create scalar index with vector field name
func TestCreateIndexWithOtherFieldName(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
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
	idxBinary, _ := entity.NewIndexBinFlat(entity.JACCARD, 64)
	err = mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinary, false)
	common.CheckErr(t, err, false, "CreateIndex failed: at most one distinct index is allowed per field")
}

func TestCreateIndexJsonField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, CollectionFieldsType: Int64FloatVecJSON, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	// create vector index on json field
	idx, _ := entity.NewIndexSCANN(entity.L2, 8, false)
	err := mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, idx, false, client.WithIndexName("json_index"))
	common.CheckErr(t, err, false, "index SCANN only supports vector data type")

	// create scalar index on json field
	type scalarIndexError struct {
		indexType entity.IndexType
		errMsg    string
	}
	inxError := []scalarIndexError{
		{entity.Bitmap, "bitmap index are only supported"},
		{entity.Inverted, "INVERTED are not supported on JSON field"},
		{entity.Sorted, "STL_SORT are only supported on numeric field"},
		{entity.Trie, "TRIE are only supported on varchar field"},
		{entity.Scalar, "create auto index on type:JSON is not supported"},
	}
	for _, ip := range inxError {
		err := mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, entity.NewScalarIndexWithType(ip.indexType), false, client.WithIndexName("json_index"))
		common.CheckErr(t, err, false, ip.errMsg)
	}

	autoIndex, _ := entity.NewIndexAUTOINDEX(entity.COSINE)
	err = mc.CreateIndex(ctx, collName, common.DefaultJSONFieldName, autoIndex, false, client.WithIndexName("json_index"))
	common.CheckErr(t, err, false, "create auto index on type:JSON is not supported")
}

func TestCreateIndexArrayField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity,
	}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	type scalarIndexError struct {
		indexType entity.IndexType
		errMsg    string
	}
	inxError := []scalarIndexError{
		{entity.Sorted, "STL_SORT are only supported on numeric field"},
		{entity.Trie, "TRIE are only supported on varchar field"},
	}

	// create scalar and vector index on array field
	vectorIdx, _ := entity.NewIndexSCANN(entity.L2, 10, false)
	for _, ip := range inxError {
		scalarIdx := entity.NewScalarIndexWithType(ip.indexType)
		collection, _ := mc.DescribeCollection(ctx, collName)
		common.PrintAllFieldNames(collName, collection.Schema)
		for _, field := range collection.Schema.Fields {
			if field.DataType == entity.FieldTypeArray {
				// create scalar index
				err := mc.CreateIndex(ctx, collName, field.Name, scalarIdx, false, client.WithIndexName("scalar_index"))
				common.CheckErr(t, err, false, ip.errMsg)
				// create vector index
				err1 := mc.CreateIndex(ctx, collName, field.Name, vectorIdx, false, client.WithIndexName("vector_index"))
				common.CheckErr(t, err1, false, "invalid parameter")
			}
		}
	}
}

func TestCreateInvertedIndexArrayField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity,
	}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: false, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	indexHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	idxErr := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexHnsw, false)
	common.CheckErr(t, idxErr, true)

	// create scalar and vector index on array field
	scalarIdx := entity.NewScalarIndexWithType(entity.Inverted)
	collection, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection.Schema.Fields {
		if field.DataType == entity.FieldTypeArray {
			// create inverted scalar index
			err := mc.CreateIndex(ctx, collName, field.Name, scalarIdx, false)
			common.CheckErr(t, err, true)
		}
	}

	// load -> search and output all vector fields
	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexAUTOINDEXSearchParam(1)
	searchRes, _ := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.COSINE, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, append(common.AllArrayFieldsName, common.DefaultFloatVecFieldName, common.DefaultFloatFieldName, common.DefaultIntFieldName))
}

func TestCreateBitmapIndexOnArrayField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity,
	}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create vector field index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// create BITMAP and SCANN index on array field
	vectorIdx, _ := entity.NewIndexSCANN(entity.L2, 10, false)
	BitmapNotSupportFiledNames := []interface{}{common.DefaultFloatArrayField, common.DefaultDoubleArrayField}
	scalarIdx := entity.NewScalarIndexWithType(entity.Bitmap)
	collection, _ := mc.DescribeCollection(ctx, collName)
	common.PrintAllFieldNames(collName, collection.Schema)
	for _, field := range collection.Schema.Fields {
		if field.DataType == entity.FieldTypeArray {
			// create scalar index
			err := mc.CreateIndex(ctx, collName, field.Name, scalarIdx, false, client.WithIndexName(field.Name+"scalar_index"))
			common.CheckErr(t, err, !common.CheckContainsValue(BitmapNotSupportFiledNames, field.Name), "bitmap index are only supported")
			// create vector index
			err1 := mc.CreateIndex(ctx, collName, field.Name, vectorIdx, false, client.WithIndexName("vector_index"))
			common.CheckErr(t, err1, false, "invalid parameter")
		}
	}

	// load -> search and output all fields
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	expr := fmt.Sprintf("%s > 10 && array_length(%s) == 100", common.DefaultIntFieldName, common.DefaultInt8ArrayField)
	searchRes, _ := mc.Search(ctx, collName, []string{}, expr, []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetInt64FloatVecArrayFieldsName(true))
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
		common.CheckErr(t, err, false, fmt.Sprintf("binary vector index does not support metric type: %v", metricType))
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
		common.CheckErr(t, errIvf, false, fmt.Sprintf("metric type %s not found or not supported, supported: [HAMMING JACCARD]", metricType),
			"binary vector index does not support metric type")
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

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	err := mc.Flush(ctx, collName, false)
	common.CheckErr(t, err, true)

	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			idx, _ := entity.NewIndexAUTOINDEX(entity.JACCARD)
			err = mc.CreateIndex(ctx, collName, fieldName, idx, false)
			common.CheckErr(t, err, true)

			// describe and check index
			indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
			expIndex := entity.NewGenericIndex(fieldName, entity.AUTOINDEX, map[string]string{"metric_type": string(entity.JACCARD)})
			common.CheckIndexResult(t, indexes, expIndex)

		} else {
			idx, _ := entity.NewIndexAUTOINDEX(entity.COSINE)
			// create index
			err = mc.CreateIndex(ctx, collName, fieldName, idx, false)
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, fieldName)
			expParams := map[string]string{
				"metric_type": string(entity.COSINE),
				"index_type":  string(entity.AUTOINDEX),
			}
			expIndex := entity.NewGenericIndex(fieldName, entity.AUTOINDEX, expParams)
			common.CheckIndexResult(t, indexes, expIndex)
		}
	}

	// load -> search and output all vector fields
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexAUTOINDEXSearchParam(1)
	searchRes, _ := mc.Search(ctx, collName, []string{}, "", common.AllVectorsFieldsName, queryVec, common.DefaultFloatVecFieldName,
		entity.COSINE, common.DefaultTopK, sp)
	common.CheckOutputFields(t, searchRes[0].Fields, common.AllVectorsFieldsName)
}

// test create default auto index on scalar fields, array and json -> error
func TestCreateAutoIndexScalarFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	// index
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			err := mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
			common.CheckErr(t, err, true)
		} else {
			err := mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
			common.CheckErr(t, err, true)
		}
	}

	// create index for all vector fields
	indexAuto, _ := entity.NewIndexAUTOINDEX(entity.L2)
	coll, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range coll.Schema.Fields {
		if SupportScalarIndexFieldType(field.DataType) {
			if field.DataType == entity.FieldTypeJSON {
				err := mc.CreateIndex(ctx, collName, field.Name, indexAuto, false, client.WithIndexName(field.Name))
				common.CheckErr(t, err, false, fmt.Sprintf("create auto index on %v field is not supported", field.DataType))
			} else {
				err := mc.CreateIndex(ctx, collName, field.Name, indexAuto, false)
				common.CheckErr(t, err, true)
				descIdx, _ := mc.DescribeIndex(ctx, collName, field.Name)
				expIdx := entity.NewGenericIndex(field.Name, entity.AUTOINDEX, indexAuto.Params())
				common.CheckIndexResult(t, descIdx, expIdx)
			}
		}
	}

	// load -> search and output all vector fields
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexAUTOINDEXSearchParam(1)
	searchRes, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckErr(t, errSearch, true)
	common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(true, false))
}

func TestCreateIndexDynamicFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	// create scalar and auto index for all vector fields
	indexAuto, _ := entity.NewIndexAUTOINDEX(entity.L2)
	for _, idx := range []entity.Index{
		indexAuto, entity.NewScalarIndex(), entity.NewScalarIndexWithType(entity.Inverted),
		entity.NewScalarIndexWithType(entity.Sorted), entity.NewScalarIndexWithType(entity.Trie), entity.NewScalarIndexWithType(entity.Bitmap),
	} {
		err := mc.CreateIndex(ctx, collName, common.DefaultDynamicFieldName, idx, false, client.WithIndexName("dynamic"))
		common.CheckErr(t, err, false, fmt.Sprintf("field %s of collection %s does not exist", common.DefaultDynamicFieldName, collName))
	}
}

// TODO https://github.com/milvus-io/milvus-sdk-go/issues/726
func TestCreateIndexSparseVector(t *testing.T) {
	t.Parallel()
	idxInverted := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_INVERTED_INDEX", map[string]string{"drop_ratio_build": "0.2", "metric_type": "IP"})
	idxWand := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_WAND", map[string]string{"drop_ratio_build": "0.3", "metric_type": "IP"})

	for _, idx := range []entity.Index{idxInverted, idxWand} {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection with all datatype
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
		}
		_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
		mc.Flush(ctx, collName, false)

		// create index
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, true)

		// describe index
		idx2, err := mc.DescribeIndex(ctx, collName, common.DefaultSparseVecFieldName)
		common.CheckErr(t, err, true)
		common.CheckIndexResult(t, idx2, idx)
	}
}

// TODO https://github.com/milvus-io/milvus-sdk-go/issues/726
func TestCreateIndexSparseVector2(t *testing.T) {
	t.Parallel()
	idxInverted1, _ := entity.NewIndexSparseInverted(entity.IP, 0.2)
	idxWand1, _ := entity.NewIndexSparseWAND(entity.IP, 0.3)
	for _, idx := range []entity.Index{idxInverted1, idxWand1} {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection with all datatype
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
		}
		_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
		mc.Flush(ctx, collName, false)

		// create index
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, true)

		// describe index
		idx2, err := mc.DescribeIndex(ctx, collName, common.DefaultSparseVecFieldName)
		expIndex := entity.NewGenericIndex(common.DefaultSparseVecFieldName, idx.IndexType(), idx.Params())
		require.EqualValues(t, expIndex, idx2[0])
		common.CheckErr(t, err, true)
		common.CheckIndexResult(t, idx2, expIndex)
	}
}

// create index on sparse vector with invalid params
func TestCreateSparseIndexInvalidParams(t *testing.T) {
	for _, indexType := range []entity.IndexType{"SPARSE_INVERTED_INDEX", "SPARSE_WAND"} {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection with all datatype
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
		}
		_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
		mc.Flush(ctx, collName, false)

		// create index with invalid metric type
		for _, mt := range common.UnsupportedSparseVecMetricsType {
			idx := entity.NewGenericIndex(common.DefaultSparseVecFieldName, indexType, map[string]string{"drop_ratio_build": "0.2", "metric_type": string(mt)})
			err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
			common.CheckErr(t, err, false, "only IP&BM25 is the supported metric type for sparse index")
		}

		// create index with invalid drop_ratio_build
		for _, drb := range []string{"a", "-0.1", "1.3"} {
			idx := entity.NewGenericIndex(common.DefaultSparseVecFieldName, indexType, map[string]string{"drop_ratio_build": drb, "metric_type": "IP"})
			err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
			common.CheckErr(t, err, false, "invalid parameter")
		}

		// create index and describe index
		idx := entity.NewGenericIndex(common.DefaultSparseVecFieldName, indexType, map[string]string{"drop_ratio_build": "0", "metric_type": "IP"})
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, true)

		descIdx, _ := mc.DescribeIndex(ctx, collName, common.DefaultSparseVecFieldName)
		common.CheckIndexResult(t, descIdx, idx)
	}
}

func TestCreateSparseIndexInvalidParams2(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
	mc.Flush(ctx, collName, false)

	// create index with invalid metric type
	for _, mt := range common.UnsupportedSparseVecMetricsType {
		idx, _ := entity.NewIndexSparseInverted(mt, 0.2)
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, false, "only IP&BM25 is the supported metric type for sparse index")

		idxWand, _ := entity.NewIndexSparseWAND(mt, 0.2)
		err = mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idxWand, false)
		common.CheckErr(t, err, false, "only IP&BM25 is the supported metric type for sparse index")
	}

	// create index with invalid drop_ratio_build
	for _, drb := range []float64{-0.3, 1.3} {
		_, err := entity.NewIndexSparseInverted(entity.IP, drb)
		common.CheckErr(t, err, false, "must be in range [0, 1)")

		_, err = entity.NewIndexSparseWAND(entity.IP, drb)
		common.CheckErr(t, err, false, "must be in range [0, 1)")
	}

	// create index and describe index
	idx, _ := entity.NewIndexSparseInverted(entity.IP, 0.1)
	err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	descIdx, _ := mc.DescribeIndex(ctx, collName, common.DefaultSparseVecFieldName)
	expIdx := entity.NewGenericIndex(common.DefaultSparseVecFieldName, idx.IndexType(), idx.Params())
	common.CheckIndexResult(t, descIdx, expIdx)
}

// create sparse unsupported index: other vector index and scalar index and auto index
func TestCreateSparseUnsupportedIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
	mc.Flush(ctx, collName, false)

	// create unsupported vector index on sparse field
	vectorIndex := append(common.GenAllFloatIndex(entity.IP))
	for _, idx := range vectorIndex {
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, false, "data type SparseFloatVector can't build with this index", "invalid parameter")
	}

	// create scalar index on sparse vector
	for _, idx := range []entity.Index{
		entity.NewScalarIndexWithType(entity.Trie),
		entity.NewScalarIndexWithType(entity.Sorted),
		entity.NewScalarIndexWithType(entity.Inverted),
		entity.NewScalarIndexWithType(entity.Bitmap),
	} {
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, false, "metric type not set for vector index")
	}
}

// create sparse auto / scalar index
func TestCreateSparseAutoIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{
		CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: 300,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithSparseVectorLen(100))
	mc.Flush(ctx, collName, false)

	// create scalar index on sparse vector
	autoIdx, _ := entity.NewIndexAUTOINDEX(entity.IP)
	for _, idx := range []entity.Index{
		entity.NewScalarIndex(),
		autoIdx,
	} {
		err := mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		common.CheckErr(t, err, true)
		idxes, err := mc.DescribeIndex(ctx, collName, common.DefaultSparseVecFieldName)
		common.CheckErr(t, err, true)
		expIndex := entity.NewGenericIndex(common.DefaultSparseVecFieldName, autoIdx.IndexType(), map[string]string{"index_type": "AUTOINDEX", "metric_type": "IP"})
		common.CheckIndexResult(t, idxes, expIndex)
		err = mc.DropIndex(ctx, collName, common.DefaultSparseVecFieldName)
		common.CheckErr(t, err, true)
	}
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
	common.CheckErr(t, err, false, fmt.Sprintf("collection %s does not exist", "haha"))
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
	common.CheckErr(t, err, false, "index HNSW only supports vector data type", "invalid parameter")

	// create scann index
	indexScann, _ := entity.NewIndexSCANN(entity.L2, 8, true)
	err = mc.CreateIndex(ctx, collName, common.DefaultFloatFieldName, indexScann, false)
	common.CheckErr(t, err, false, "index SCANN only supports vector data type", "invalid parameter")
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
			fmt.Sprintf("float vector index does not support metric type: %s", mt))
	}

	// invalid flat metric type jaccard for flat index
	idx, _ := entity.NewIndexFlat(entity.JACCARD)
	errMetricType := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, errMetricType, false,
		"float vector index does not support metric type: JACCARD")
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
	cp := CollectionParams{
		CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllVectors,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
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
	common.CheckErr(t, errDescribe, false, "index not found")
	require.Nil(t, indexes)

	// create IP index
	ipIdx, _ := entity.NewIndexHNSW(entity.IP, 8, 96)
	err2 := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, ipIdx, false)
	common.CheckErr(t, err2, true)

	// describe index
	ipIndexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	require.EqualValues(t, entity.NewGenericIndex(common.DefaultFloatVecFieldName, ipIdx.IndexType(), ipIdx.Params()), ipIndexes[0])

	// describe collection
	t.Log("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/385")
	collection2, _ := mc.DescribeCollection(ctx, collName)
	for _, field := range collection2.Schema.Fields {
		log.Printf("field name: %v, field TypeParams: %v, field IndexParams: %v)", field.Name, field.TypeParams, field.IndexParams)
	}
}
