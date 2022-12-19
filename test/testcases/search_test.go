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

func TestSearch(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchResult, errSearch := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		//[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, true)
	common.CheckOutputFields(t, searchResult[0].Fields, []string{common.DefaultFloatFieldName})
	common.CheckSearchResult(t, searchResult, common.DefaultNq, common.DefaultTopK)
}

// test search collection not exist
func TestSearchCollectionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// search vector
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	_, errSearch := mc.Search(
		ctx, "collName",
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		//[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, false, "can't find collection: collName")
}

// test search empty collection -> return empty
func TestSearchEmptyCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// empty collection
	collName := createDefaultCollection(ctx, t, mc, false)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchRes, _ := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		//[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	require.Len(t, searchRes, 0)
}

// test search with partition names []string{}, []string{""}
func TestSearchEmptyPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false)
	_, vecColumnDefault, vecColumnPartition := createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load with not exist partition names
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search with empty partition name []string{""}
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	_, errSearch := mc.Search(
		ctx, collName,
		[]string{""},
		"",
		[]string{common.DefaultFloatFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, false, "Partition name should not be empty")

	// search with empty partition names slice []string{}
	searchResult, _ := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultFloatFieldName},
		[]entity.Vector{
			entity.FloatVector(vecColumnDefault.VectorColumn.Data()[0]),
			entity.FloatVector(vecColumnPartition.VectorColumn.Data()[0]),
		},
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)

	// check search result contains search vector, which from all partitions
	common.CheckSearchResult(t, searchResult, 2, common.DefaultTopK)
	log.Println(searchResult[0].IDs.(*entity.ColumnInt64).Data())
	log.Println(searchResult[1].IDs.(*entity.ColumnInt64).Data())
	require.Contains(t, searchResult[0].IDs.(*entity.ColumnInt64).Data(), vecColumnDefault.IdsColumn.(*entity.ColumnInt64).Data()[0])
	require.Contains(t, searchResult[1].IDs.(*entity.ColumnInt64).Data(), vecColumnPartition.IdsColumn.(*entity.ColumnInt64).Data()[0])
}

// test search with an not existed partition -> error
func TestSearchPartitionNotExist(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	type notExistPartitions []string

	// search partitions not exist, part exist
	partitionsNotExist := []notExistPartitions{[]string{"new"}, []string{"new", common.DefaultPartition}}

	for _, partitions := range partitionsNotExist {
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		_, errSearch := mc.Search(
			ctx, collName,
			partitions,
			"",
			[]string{common.DefaultFloatFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearch, false, "partition name new not found")
	}
}

// test search single partition and multi partitions
func TestSearchPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false)
	partitionName, vecColumnDefault, vecColumnPartition := createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load with not exist partition names
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search single partition
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchSingleRes, _ := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		[]entity.Vector{
			entity.FloatVector(vecColumnDefault.VectorColumn.Data()[0]),
			entity.FloatVector(vecColumnPartition.VectorColumn.Data()[0]),
		},
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	// check search result contains search vector, which from all partitions
	common.CheckSearchResult(t, searchSingleRes, 2, common.DefaultTopK)
	require.Contains(t, searchSingleRes[0].IDs.(*entity.ColumnInt64).Data(), vecColumnDefault.IdsColumn.(*entity.ColumnInt64).Data()[0])

	// search multi partitions
	searchMultiRes, _ := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition, partitionName},
		"",
		[]string{common.DefaultFloatFieldName},
		[]entity.Vector{
			entity.FloatVector(vecColumnDefault.VectorColumn.Data()[0]),
			entity.FloatVector(vecColumnPartition.VectorColumn.Data()[0]),
		},
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckSearchResult(t, searchMultiRes, 2, common.DefaultTopK)
	require.Contains(t, searchMultiRes[0].IDs.(*entity.ColumnInt64).Data(), vecColumnDefault.IdsColumn.(*entity.ColumnInt64).Data()[0])
	require.Contains(t, searchMultiRes[1].IDs.(*entity.ColumnInt64).Data(), vecColumnPartition.IdsColumn.(*entity.ColumnInt64).Data()[0])
}

// test search empty output fields []string{} -> [], []string{""}
func TestSearchEmptyOutputFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector output fields []string{} -> []
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchResPkOutput, errSearch := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, true)
	// todo Issue https://github.com/milvus-io/milvus/issues/21112
	common.CheckOutputFields(t, searchResPkOutput[0].Fields, []string{})
	common.CheckSearchResult(t, searchResPkOutput, common.DefaultNq, common.DefaultTopK)

	// search vector output fields []string{""}
	_, errSearchExist := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{""},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchExist, false, "exist")
}

// test search output fields not exist -> error
func TestSearchNotExistOutputFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	type notExistOutputFields []string

	// search vector output fields not exist, part exist
	outputFields := []notExistOutputFields{[]string{"field"}, []string{"fields", common.DefaultFloatFieldName}}
	for _, fields := range outputFields {
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		_, errSearch := mc.Search(
			ctx, collName,
			[]string{},
			"",
			fields,
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearch, false, "not exist")
	}
}

// test search output fields only pk
func TestSearchOutputFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createVarcharCollectionWithDataIndex(ctx, t, mc, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector output fields not exist
	sp, _ := entity.NewIndexBinIvfFlatSearchParam(64)
	searchRes, _ := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultVarcharFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector),
		common.DefaultBinaryVecFieldName,
		entity.JACCARD,
		common.DefaultTopK,
		sp,
	)
	common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultVarcharFieldName})
	common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)

	// search output varchar fields and varchar
	_, errSearch := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultVarcharFieldName, common.DefaultBinaryVecFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector),
		common.DefaultBinaryVecFieldName,
		entity.JACCARD,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, false, "search doesn't support vector field as output_fields")
}

// test search output all scalar fields
func TestSearchOutputAllScalarFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create and insert
	collName, _ := createCollectionAllFields(ctx, t, mc, common.DefaultNb, 0)
	mc.Flush(ctx, collName, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, "floatVec", idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector output all scalar fields
	allScalarFields := []string{"int64", "bool", "int8", "int16", "int32", "float", "double", "varchar"}
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchRes, _ := mc.Search(
		ctx, collName,
		[]string{},
		"",
		allScalarFields,
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		"floatVec",
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckOutputFields(t, searchRes[0].Fields, allScalarFields)
}

// test search with invalid vector field name: not exist; non-vector field, empty fiend name -> error
func TestSearchInvalidVectorField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	type invalidVectorFieldStruct struct {
		vectorField string
		errMsg      string
	}

	invalidVectorFields := []invalidVectorFieldStruct{
		// not exist field
		{vectorField: common.DefaultBinaryVecFieldName, errMsg: fmt.Sprintf("failed to get field schema by name: fieldName(%s) not found", common.DefaultBinaryVecFieldName)},

		// non-vector field
		{vectorField: common.DefaultIntFieldName, errMsg: fmt.Sprintf("failed to create query plan: field (%s) to search is not of vector data type", common.DefaultIntFieldName)},

		// empty field name
		{vectorField: "", errMsg: "failed to get field schema by name: fieldName() not found"},
	}

	sp, _ := entity.NewIndexHNSWSearchParam(74)
	for _, invalidVectorField := range invalidVectorFields {
		_, errSearchNotExist := mc.Search(
			ctx, collName,
			[]string{},
			"",
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			invalidVectorField.vectorField,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchNotExist, false, invalidVectorField.errMsg)
	}
}

// test search with invalid vectors
// TODO Issue https://github.com/milvus-io/milvus-sdk-go/issues/377
func TestSearchInvalidVectors(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	type invalidVectorsStruct struct {
		vectors []entity.Vector
		errMsg  string
	}

	invalidVectors := []invalidVectorsStruct{
		// dim not match
		{vectors: common.GenSearchVectors(common.DefaultNq, 64, entity.FieldTypeFloatVector), errMsg: "vector dimension mismatch"},

		// vector type not match
		{vectors: common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector), errMsg: "fail to search on all shard leaders"},

		// empty vectors
		{vectors: []entity.Vector{}, errMsg: "nq [0] is invalid"},
		{vectors: []entity.Vector{entity.FloatVector{}}, errMsg: "fail to search on all shard leaders"},
	}

	sp, _ := entity.NewIndexHNSWSearchParam(74)
	for _, invalidVector := range invalidVectors {
		// search vectors empty slice
		_, errSearchEmpty := mc.Search(
			ctx, collName,
			[]string{},
			"",
			[]string{common.DefaultIntFieldName},
			invalidVector.vectors,
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, false, invalidVector.errMsg)
	}
}

// test search metric type isn't the same with index metric type
func TestSearchNotMatchMetricType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	sp, _ := entity.NewIndexHNSWSearchParam(74)
	_, errSearchEmpty := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultIntFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.IP,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchEmpty, false, "Metric type of field index isn't the same with search info")
}

// test search with invalid topK -> error
func TestSearchInvalidTopK(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create ivf sq8 index
	idx, _ := entity.NewIndexIvfFlat(entity.L2, 128)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))
	common.CheckErr(t, err, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	invalidTopKs := []int{-1, 0, 16385}

	sp, _ := entity.NewIndexIvfSQ8SearchParam(64)

	for _, invalidTopK := range invalidTopKs {
		_, errSearchEmpty := mc.Search(
			ctx, collName,
			[]string{},
			"",
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			invalidTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, false, "should be in range [1, 16384]")
	}
}

// test search with invalid search params
func TestSearchInvalidSearchParams(t *testing.T) {

	// ivf flat search params nlist [1, 65536], nprobe [1, nlist]
	invalidNprobe := []int{-1, 0, 65537}
	for _, nprobe := range invalidNprobe {
		_, errIvfFlat := entity.NewIndexIvfFlatSearchParam(nprobe)
		common.CheckErr(t, errIvfFlat, false, "nprobe not valid")
	}

	// ivf sq8 search param
	for _, nprobe := range invalidNprobe {
		_, errIvfSq8 := entity.NewIndexIvfSQ8SearchParam(nprobe)
		common.CheckErr(t, errIvfSq8, false, "nprobe not valid")
	}

	// ivf pq search param
	for _, nprobe := range invalidNprobe {
		_, errIvfPq := entity.NewIndexIvfPQSearchParam(nprobe)
		common.CheckErr(t, errIvfPq, false, "nprobe not valid")
	}

	// hnsw search params ef [top_k, 32768]
	invalidEfs := []int{-1, 0, 32769}
	for _, invalidEf := range invalidEfs {
		_, errHnsw := entity.NewIndexHNSWSearchParam(invalidEf)
		common.CheckErr(t, errHnsw, false, "ef not valid")
	}

	// TODO annoy {-1} ∪ [top_k, top_k × n_trees]

	// bin ivf flat
	for _, nprobe := range invalidNprobe {
		_, errBinIvfFlat := entity.NewIndexBinIvfFlatSearchParam(nprobe)
		common.CheckErr(t, errBinIvfFlat, false, "nprobe not valid")
	}
}

// search with index hnsw search param ef < topK -> error
func TestSearchTopKHnsw(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// ef [top_k, 32768]
	sp, _ := entity.NewIndexHNSWSearchParam(7)
	_, errSearchEmpty := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultIntFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchEmpty, false, "Param 'ef'(7) is not in range [10, 32768] err")
}

// test search invalid annoy search params about topK
func TestSearchTopKAnnoy(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/378")
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// create annoy index
	idx, _ := entity.NewIndexANNOY(entity.L2, 56)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search_k {-1} ∪ [top_k, top_k × n_trees]
	invalidSearchK := []int{-5, 9, 1000}
	for _, searchK := range invalidSearchK {
		sp, _ := entity.NewIndexANNOYSearchParam(searchK)
		_, errSearchEmpty := mc.Search(
			ctx, collName,
			[]string{},
			"",
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, false, "Param is not in range err")
	}
}

// test search params mismatch index type, hnsw index and ivf sq8 search param
func TestSearchSearchParamsMismatchIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// ef [top_k, 32768]
	sp, _ := entity.NewIndexIvfSQ8SearchParam(64)
	_, errSearchEmpty := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{common.DefaultIntFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchEmpty, false, "Param 'ef' not exist err")
}

// test search expr
func TestSearchExpr(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search vector with expr: int64 < 1000
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchResult, errSearch := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		fmt.Sprintf("%s < 1000", common.DefaultIntFieldName),
		[]string{common.DefaultFloatFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearch, true)
	// check search ids less than 1000
	common.CheckSearchResult(t, searchResult, common.DefaultNq, common.DefaultTopK)
	for _, res := range searchResult {
		for _, id := range res.IDs.(*entity.ColumnInt64).Data() {
			require.Less(t, id, int64(1000))
		}
	}

	// search vector with expr: float in [1.0]
	searchResult2, errSearch2 := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		fmt.Sprintf("%s in [1.0]", common.DefaultFloatFieldName),
		[]string{common.DefaultFloatFieldName},
		common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		2,
		sp,
	)
	common.CheckErr(t, errSearch2, true)
	// check search ids equal to 1
	common.CheckSearchResult(t, searchResult2, 1, 1)
	require.Equal(t, searchResult2[0].IDs.(*entity.ColumnInt64).Data()[0], int64(1))
}

// test search invalid expr
func TestSearchInvalidExpr(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	sp, _ := entity.NewIndexHNSWSearchParam(74)

	// invalid expr and error message
	type invalidExprStruct struct {
		expr   string
		errMsg string
	}
	invalidExpr := []invalidExprStruct{
		{expr: "id in [0]", errMsg: "fieldName(id) not found"},               // not exist field
		{expr: "int64 in not [0]", errMsg: "cannot parse expression"},        // wrong term expr keyword
		{expr: "int64 < floatVec", errMsg: "unsupported datatype"},           // unsupported compare field
		{expr: "floatVec in [0]", errMsg: "cannot be casted to FloatVector"}, // value and field type mismatch
	}

	for _, exprStruct := range invalidExpr {
		_, errSearchEmpty := mc.Search(
			ctx, collName,
			[]string{},
			exprStruct.expr,
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, false, exprStruct.errMsg)
	}
}

// TODO offset and limit
// TODO consistency level
// TODO WithGuaranteeTimestamp
