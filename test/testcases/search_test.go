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
	common.CheckErr(t, errSearch, false, "collection not found")
}

// test search empty collection -> return empty
func TestSearchEmptyCollection(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamicField := range []bool{true, false} {
		// empty collection
		collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithEnableDynamicSchema(enableDynamicField))

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
}

// test search with partition names []string{}, []string{""}
func TestSearchEmptyPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamicField := range []bool{true, false} {
		// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
		collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithEnableDynamicSchema(enableDynamicField))
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
		vecDefaultData := vecColumnDefault.VectorColumn.(*entity.ColumnFloatVector).Data()[0]
		vecPartitionData := vecColumnPartition.VectorColumn.(*entity.ColumnFloatVector).Data()[0]
		searchResult, _ := mc.Search(
			ctx, collName,
			[]string{},
			"",
			[]string{common.DefaultFloatFieldName},
			[]entity.Vector{
				entity.FloatVector(vecDefaultData),
				entity.FloatVector(vecPartitionData),
			},
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)

		// check search result contains search vector, which from all partitions
		nq0IDs := searchResult[0].IDs.(*entity.ColumnInt64).Data()
		nq1IDs := searchResult[1].IDs.(*entity.ColumnInt64).Data()
		common.CheckSearchResult(t, searchResult, 2, common.DefaultTopK)
		log.Println(nq0IDs)
		log.Println(nq1IDs)
		require.Contains(t, nq0IDs, vecColumnDefault.IdsColumn.(*entity.ColumnInt64).Data()[0])
		require.Contains(t, nq1IDs, vecColumnPartition.IdsColumn.(*entity.ColumnInt64).Data()[0])
	}
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
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, vecColumnDefault, vecColumnPartition := createInsertTwoPartitions(ctx, t, mc, collName, 500)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// load with not exist partition names
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	vecDefaultData := vecColumnDefault.VectorColumn.(*entity.ColumnFloatVector).Data()[0]
	vecPartitionData := vecColumnPartition.VectorColumn.(*entity.ColumnFloatVector).Data()[0]

	// search single partition
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchSingleRes, _ := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		[]entity.Vector{
			entity.FloatVector(vecDefaultData),
			entity.FloatVector(vecPartitionData),
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
			entity.FloatVector(vecDefaultData),
			entity.FloatVector(vecPartitionData),
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
	t.Skip("https://github.com/milvus-io/milvus/issues/28465")
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamic := range []bool{true, false} {
		// create collection with data
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true, client.WithEnableDynamicSchema(enableDynamic))

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

		//if enableDynamic {
		//	common.CheckErr(t, errSearchExist, true)
		//	common.CheckOutputFields(t, sp1[0].Fields, []string{""})
		//} else {
		common.CheckErr(t, errSearchExist, false, "not exist")
		//}
		common.CheckSearchResult(t, searchResPkOutput, common.DefaultNq, common.DefaultTopK)
	}
}

// test search output fields not exist -> output existed fields
func TestSearchNotExistOutputFields(t *testing.T) {
	t.Skip("https://github.com/milvus-io/milvus/issues/28465")
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamic := range []bool{false} {
		// create collection with data
		collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true, client.WithEnableDynamicSchema(enableDynamic))

		// load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		type notExistOutputFields []string

		// search vector output fields not exist, part exist
		outputFields := []notExistOutputFields{[]string{"aaa"}, []string{"fields", common.DefaultFloatFieldName},
			[]string{"fields", "*"}}
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		for _, fields := range outputFields {
			log.Println(fields)
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
			common.CheckErr(t, errSearch, true)
		}
	}
}

// test search output fields only pk
func TestSearchOutputFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamic := range []bool{true, false} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: VarcharBinaryVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: enableDynamic}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		// index
		idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName(""))
		common.CheckErr(t, err, true)

		// load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// search vector output fields not exist
		outputFields := []string{common.DefaultVarcharFieldName, common.DefaultBinaryVecFieldName}
		if enableDynamic {
			outputFields = append(outputFields, common.DefaultDynamicFieldName)
		}
		sp, _ := entity.NewIndexBinIvfFlatSearchParam(64)
		searchRes, _ := mc.Search(
			ctx, collName,
			[]string{},
			"",
			outputFields,
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector),
			common.DefaultBinaryVecFieldName,
			entity.JACCARD,
			common.DefaultTopK,
			sp,
		)
		common.CheckOutputFields(t, searchRes[0].Fields, outputFields)
		common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
	}
}

// test search output all * fields when enable dynamic and insert dynamic column data
func TestSearchOutputAllFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamic := range []bool{false, true} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: enableDynamic}
		_, _ = insertData(ctx, t, mc, dp)
		_ = mc.Flush(ctx, collName, false)

		// create index
		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, "floatVec", idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// search vector output all scalar fields
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		allFields := []string{"int64", "bool", "int8", "int16", "int32", "float", "double", "varchar", "json", "floatVec"}
		allFields = append(allFields, common.AllArrayFieldsName...)

		if enableDynamic {
			allFields = append(allFields, common.DefaultDynamicFieldName)
		}
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		searchRes, _ := mc.Search(ctx, collName, []string{},
			"",
			[]string{"*"},
			queryVec,
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckOutputFields(t, searchRes[0].Fields, allFields)

		// search with output * fields
		if enableDynamic {
			//search output [*, a] fields -> output all fields, no a field
			_, errNotExist := mc.Search(ctx, collName, []string{}, "", []string{"*", "a"}, queryVec,
				common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, sp)
			common.CheckErr(t, errNotExist, true)
			common.CheckOutputFields(t, searchRes[0].Fields, allFields)

			//search output [*, dynamicNumber] fields -> -> output all fields, $meta replace by dynamicNumber
			searchRes, _ = mc.Search(ctx, collName, []string{}, "", []string{"*", common.DefaultDynamicNumberField},
				queryVec, common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, sp)
			common.CheckOutputFields(t, searchRes[0].Fields, append(allFields, common.DefaultDynamicNumberField))

			///search output [*, dynamicNumber] fields -> -> output all fields, $meta replace by dynamicNumber
			searchRes, _ = mc.Search(ctx, collName, []string{}, "", []string{common.DefaultDynamicNumberField},
				queryVec, common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, sp)
			common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultDynamicNumberField})
		}
	}
}

// test search with invalid vector field name: not exist; non-vector field, empty fiend name, json and dynamic field -> error
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
		errNil      bool
		errMsg      string
	}

	invalidVectorFields := []invalidVectorFieldStruct{
		// not exist field
		{vectorField: common.DefaultBinaryVecFieldName, errNil: false, errMsg: fmt.Sprintf("failed to get field schema by name: fieldName(%s) not found", common.DefaultBinaryVecFieldName)},

		// non-vector field
		{vectorField: common.DefaultIntFieldName, errNil: false, errMsg: fmt.Sprintf("failed to create query plan: field (%s) to search is not of vector data type", common.DefaultIntFieldName)},

		// json field
		{vectorField: common.DefaultJSONFieldName, errNil: false, errMsg: fmt.Sprintf("failed to get field schema by name: fieldName(%s) not found", common.DefaultJSONFieldName)},

		// dynamic field
		{vectorField: common.DefaultDynamicFieldName, errNil: false, errMsg: fmt.Sprintf("failed to get field schema by name: fieldName(%s) not found", common.DefaultDynamicFieldName)},

		// allows empty vector field name
		{vectorField: "", errNil: true, errMsg: ""},
	}

	sp, _ := entity.NewIndexHNSWSearchParam(74)
	for _, invalidVectorField := range invalidVectorFields {
		t.Run(invalidVectorField.vectorField, func(t *testing.T) {
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
			common.CheckErr(t, errSearchNotExist, invalidVectorField.errNil, invalidVectorField.errMsg)
		})
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
		{vectors: common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector), errMsg: "failed to search"},

		// empty vectors
		{vectors: []entity.Vector{}, errMsg: "nq [0] is invalid"},
		{vectors: []entity.Vector{entity.FloatVector{}}, errMsg: "failed to search"},
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
	common.CheckErr(t, errSearchEmpty, false, "invalid parameter")
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
		common.CheckErr(t, errIvfFlat, false, "nprobe has to be in range [1, 65536]")
	}

	// ivf sq8 search param
	for _, nprobe := range invalidNprobe {
		_, errIvfSq8 := entity.NewIndexIvfSQ8SearchParam(nprobe)
		log.Println(nprobe)
		common.CheckErr(t, errIvfSq8, false, "nprobe has to be in range [1, 65536]")
	}

	// ivf pq search param
	for _, nprobe := range invalidNprobe {
		_, errIvfPq := entity.NewIndexIvfPQSearchParam(nprobe)
		common.CheckErr(t, errIvfPq, false, "nprobe has to be in range [1, 65536]")
	}

	// hnsw search params ef [top_k, 32768]
	invalidEfs := []int{-1, 0, 32769}
	for _, invalidEf := range invalidEfs {
		_, errHnsw := entity.NewIndexHNSWSearchParam(invalidEf)
		common.CheckErr(t, errHnsw, false, "ef has to be in range [1, 32768]")
	}

	// bin ivf flat
	for _, nprobe := range invalidNprobe {
		_, errBinIvfFlat := entity.NewIndexBinIvfFlatSearchParam(nprobe)
		common.CheckErr(t, errBinIvfFlat, false, "nprobe has to be in range [1, 65536]")
	}

	// scann index invalid nprobe
	for _, nprobe := range invalidNprobe {
		_, errScann := entity.NewIndexSCANNSearchParam(nprobe, 100)
		common.CheckErr(t, errScann, false, "nprobe has to be in range [1, 65536]")
	}

	_, errScann := entity.NewIndexSCANNSearchParam(16, 0)
	common.CheckErr(t, errScann, false, "reorder_k has to be in range [1, 9223372036854775807]")
}

// search with index hnsw search param ef < topK -> error
func TestSearchEfHnsw(t *testing.T) {
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
	common.CheckErr(t, errSearchEmpty, false, "ef(7) should be larger than k(10)")
}

// test search params mismatch index type, hnsw index and ivf sq8 search param -> search with default hnsw params, ef=topK
func TestSearchSearchParamsMismatchIndex(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data and create hnsw index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// ef [top_k, 32768]
	sp, _ := entity.NewIndexIvfSQ8SearchParam(64)
	resSearch, errSearch := mc.Search(
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
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, resSearch, common.DefaultNq, common.DefaultTopK)
}

// test search with valid expression
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
		fmt.Sprintf("%s < 1000", common.DefaultFloatFieldName),
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

// test search with invalid expression
func TestSearchInvalidExpr(t *testing.T) {
	t.Parallel()

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search with invalid expr
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	for _, exprStruct := range common.InvalidExpressions {
		_, errSearchEmpty := mc.Search(
			ctx, collName,
			[]string{},
			exprStruct.Expr,
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, exprStruct.ErrNil, exprStruct.ErrMsg)
	}
}

func TestSearchJsonFieldExpr(t *testing.T) {
	t.Parallel()

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, dynamicField := range []bool{false} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: dynamicField,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: dynamicField}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		exprs := []string{
			"",
			fmt.Sprintf("exists %s['number'] ", common.DefaultJSONFieldName),   // exists
			"json[\"number\"] > 1 and json[\"number\"] < 1000",                 // > and
			fmt.Sprintf("%s[\"number\"] > 10", common.DefaultJSONFieldName),    // number >
			fmt.Sprintf("%s[\"number\"] < 2000", common.DefaultJSONFieldName),  // number <
			fmt.Sprintf("%s[\"bool\"] != true", common.DefaultJSONFieldName),   // bool !=
			fmt.Sprintf("%s[\"bool\"] == False", common.DefaultJSONFieldName),  // bool ==
			fmt.Sprintf("%s[\"bool\"] in [true]", common.DefaultJSONFieldName), // bool in
			fmt.Sprintf("%s[\"string\"] >= '1' ", common.DefaultJSONFieldName), // string >=
			fmt.Sprintf("%s['list'][0] > 200", common.DefaultJSONFieldName),    // list filter
		}

		// search with jsonField expr key datatype and json data type mismatch
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		for _, expr := range exprs {
			log.Printf("search expr: %s", expr)
			searchRes, errSearchEmpty := mc.Search(
				ctx, collName,
				[]string{},
				expr,
				[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName},
				common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
				common.DefaultFloatVecFieldName,
				entity.L2,
				common.DefaultTopK,
				sp,
			)
			common.CheckErr(t, errSearchEmpty, true)
			common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultIntFieldName, common.DefaultJSONFieldName})
			common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
		}
	}
}

// search dynamic field with expr
func TestSearchDynamicFieldExpr(t *testing.T) {
	t.Parallel()

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, withRows := range []bool{true, false} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		exprs := []string{
			"",
			"exists dynamicNumber", // exist without dynamic fieldName
			fmt.Sprintf("exists %s[\"dynamicNumber\"]", common.DefaultDynamicFieldName), // exist with fieldName
			fmt.Sprintf("%s[\"dynamicNumber\"] > 10", common.DefaultDynamicFieldName),   // int expr with fieldName
			fmt.Sprintf("%s[\"dynamicBool\"] == true", common.DefaultDynamicFieldName),  // bool with fieldName
			"dynamicBool == False", // bool without fieldName
			fmt.Sprintf("%s['dynamicString'] == '1'", common.DefaultDynamicFieldName), // string with fieldName
			"dynamicString != \"2\" ", // string without fieldName
		}

		// search with jsonField expr key datatype and json data type mismatch
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		for _, expr := range exprs {
			log.Print(expr)
			searchRes, errSearchEmpty := mc.Search(
				ctx, collName,
				[]string{},
				expr,
				[]string{common.DefaultIntFieldName, "dynamicNumber", "number"},
				common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
				common.DefaultFloatVecFieldName,
				entity.L2,
				common.DefaultTopK,
				sp,
			)
			common.CheckErr(t, errSearchEmpty, true)
			common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultIntFieldName, "dynamicNumber", "number"})
			if expr == "$meta['dynamicString'] == '1'" {
				common.CheckSearchResult(t, searchRes, common.DefaultNq, 1)
			} else {
				common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
			}
		}

		// search with expr filter number and, &&, or, ||
		exprs2 := []string{
			"dynamicNumber > 1 and dynamicNumber <= 999", // int expr without fieldName
			fmt.Sprintf("%s['dynamicNumber'] > 1 && %s['dynamicNumber'] < 1000", common.DefaultDynamicFieldName, common.DefaultDynamicFieldName),
			"dynamicNumber < 888 || dynamicNumber < 1000",
			fmt.Sprintf("%s['dynamicNumber'] < 888 or %s['dynamicNumber'] < 1000", common.DefaultDynamicFieldName, common.DefaultDynamicFieldName),
			fmt.Sprintf("%s[\"dynamicNumber\"] < 1000", common.DefaultDynamicFieldName), // int expr with fieldName
		}

		// search
		for _, expr := range exprs2 {
			log.Print(expr)
			searchRes, errSearchEmpty := mc.Search(
				ctx, collName,
				[]string{},
				expr,
				[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicFieldName, "dynamicNumber", "number"},
				common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
				common.DefaultFloatVecFieldName,
				entity.L2,
				common.DefaultTopK,
				sp,
			)
			common.CheckErr(t, errSearchEmpty, true)
			common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicFieldName, "dynamicNumber", "number"})
			for _, res := range searchRes {
				for _, id := range res.IDs.(*entity.ColumnInt64).Data() {
					require.Less(t, id, int64(1000))
				}
			}
		}
	}
}

func TestSearchArrayFieldExpr(t *testing.T) {
	t.Parallel()

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	var capacity int64 = common.TestCapacity

	for _, withRows := range []bool{true, false} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity}
		collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

		// prepare and insert data
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows}
		_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))
		mc.Flush(ctx, collName, false)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		exprs := []string{
			fmt.Sprintf("%s[0] == false", common.DefaultBoolArrayField),                            // array[0] ==
			fmt.Sprintf("%s[0] > 0", common.DefaultInt64ArrayField),                                // array[0] >
			fmt.Sprintf("json_contains (%s, %d)", common.DefaultInt16ArrayField, capacity),         // json_contains
			fmt.Sprintf("array_contains (%s, %d)", common.DefaultInt16ArrayField, capacity),        // array_contains
			fmt.Sprintf("json_contains_all (%s, [90, 91])", common.DefaultInt64ArrayField),         // json_contains_all
			fmt.Sprintf("array_contains_all (%s, [90, 91])", common.DefaultInt64ArrayField),        // array_contains_all
			fmt.Sprintf("array_contains_any (%s, [0, 100, 10000])", common.DefaultFloatArrayField), // array_contains_any
			fmt.Sprintf("json_contains_any (%s, [0, 100, 10])", common.DefaultFloatArrayField),     // json_contains_any
			fmt.Sprintf("array_length(%s) == %d", common.DefaultDoubleArrayField, capacity),        // array_length
		}

		// search with jsonField expr key datatype and json data type mismatch
		sp, _ := entity.NewIndexHNSWSearchParam(74)
		for _, expr := range exprs {
			log.Printf("search expr: %s", expr)
			searchRes, errSearchEmpty := mc.Search(
				ctx, collName, []string{},
				expr, common.AllArrayFieldsName,
				common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
				common.DefaultFloatVecFieldName, entity.L2,
				common.DefaultTopK, sp,
			)
			common.CheckErr(t, errSearchEmpty, true)
			common.CheckOutputFields(t, searchRes[0].Fields, common.AllArrayFieldsName)
			common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
		}

		// search hits empty
		searchRes, errSearchEmpty := mc.Search(
			ctx, collName, []string{},
			fmt.Sprintf("array_contains (%s, 1000000)", common.DefaultInt32ArrayField),
			common.AllArrayFieldsName,
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearchEmpty, true)
		require.Empty(t, searchRes)
	}
}

// search with index scann search param ef < topK -> error
func TestSearchInvalidScannReorderK(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp)

	// flush
	mc.Flush(ctx, collName, false)

	// create scann index
	indexScann, _ := entity.NewIndexSCANN(entity.L2, 16, false)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexScann, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	log.Println(indexes)
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.SCANN, indexScann.Params())
	common.CheckIndexResult(t, indexes, expIndex)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search with invalid reorder_k < topk
	queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	spInvalid, _ := entity.NewIndexSCANNSearchParam(8, common.DefaultTopK-1)
	_, errInvalid := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName}, queryVec,
		common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, spInvalid)
	common.CheckErr(t, errInvalid, false,
		fmt.Sprintf("reorder_k(%d) should be larger than k(%d)", common.DefaultTopK-1, common.DefaultTopK))

	// valid scann index search reorder_k
	sp, _ := entity.NewIndexSCANNSearchParam(8, 20)
	resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName}, queryVec,
		common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, sp)
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
}

// test search with scann index params: with_raw_data and metrics_type [L2, IP, COSINE]
func TestSearchScannAllMetricsWithRawData(t *testing.T) {
	t.Parallel()
	for _, with_raw_data := range []bool{true, false} {
		for _, metricType := range []entity.MetricType{entity.L2, entity.IP, entity.COSINE} {
			ctx := createContext(t, time.Second*common.DefaultTimeout)
			// connect
			mc := createMilvusClient(ctx, t)

			// create collection
			cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
				ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
			collName := createCollection(ctx, t, mc, cp)

			// insert
			dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
				start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
			_, _ = insertData(ctx, t, mc, dp)
			mc.Flush(ctx, collName, false)

			// create scann index
			indexScann, _ := entity.NewIndexSCANN(metricType, 16, with_raw_data)
			err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexScann, false)
			common.CheckErr(t, err, true)

			// describe index
			indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
			expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.SCANN, indexScann.Params())
			common.CheckIndexResult(t, indexes, expIndex)

			// load collection
			errLoad := mc.LoadCollection(ctx, collName, false)
			common.CheckErr(t, errLoad, true)

			// search and output all fields
			sp, _ := entity.NewIndexSCANNSearchParam(8, 20)
			resSearch, errSearch := mc.Search(
				ctx, collName,
				[]string{},
				"",
				[]string{"*"},
				common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector),
				common.DefaultFloatVecFieldName,
				metricType,
				common.DefaultTopK,
				sp,
			)
			common.CheckErr(t, errSearch, true)
			common.CheckOutputFields(t, resSearch[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName,
				common.DefaultJSONFieldName, common.DefaultFloatVecFieldName, common.DefaultDynamicFieldName})
			common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
		}
	}
}

// test range search with scann index
func TestRangeSearchScannL2(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp)

	// create scann index
	indexScann, _ := entity.NewIndexSCANN(entity.L2, 16, false)
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexScann, false)
	common.CheckErr(t, err, true)

	// describe index
	indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.SCANN, indexScann.Params())
	common.CheckIndexResult(t, indexes, expIndex)

	// load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// search output all fields
	queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexSCANNSearchParam(8, 20)
	sp.AddRadius(20)
	sp.AddRangeFilter(15)
	resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)

	// verify error nil, output all fields, range score
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
	common.CheckOutputFields(t, resSearch[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName,
		common.DefaultJSONFieldName, common.DefaultFloatVecFieldName, common.DefaultDynamicFieldName})
	for _, s := range resSearch[0].Scores {
		require.GreaterOrEqual(t, s, float32(15.0))
		require.Less(t, s, float32(20.0))
	}

	// invalid range search: radius < range filter
	sp.AddRadius(15)
	sp.AddRangeFilter(20)
	_, errRange := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
		entity.L2, common.DefaultTopK, sp)
	common.CheckErr(t, errRange, false, "range_filter must be less than radius for L2/HAMMING/JACCARD")
}

// test range search with scann index and IP COSINE metric type
func TestRangeSearchScannIPCosine(t *testing.T) {
	t.Parallel()
	for _, metricType := range []entity.MetricType{entity.IP, entity.COSINE} {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		// create scann index
		indexScann, _ := entity.NewIndexSCANN(metricType, 16, false)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexScann, false)
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
		expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.SCANN, indexScann.Params())
		common.CheckIndexResult(t, indexes, expIndex)

		// load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// range search filter distance and output all fields
		queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexSCANNSearchParam(8, 20)
		sp.AddRadius(0)
		sp.AddRangeFilter(100)
		resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
			metricType, common.DefaultTopK, sp)

		// verify error nil, output all fields, range score
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
		common.CheckOutputFields(t, resSearch[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName,
			common.DefaultJSONFieldName, common.DefaultFloatVecFieldName, common.DefaultDynamicFieldName})
		for _, s := range resSearch[0].Scores {
			require.GreaterOrEqual(t, s, float32(0))
			require.Less(t, s, float32(100))
		}

		// invalid range search: radius > range filter
		sp.AddRadius(20)
		sp.AddRangeFilter(10)
		_, errRange := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultFloatVecFieldName,
			metricType, common.DefaultTopK, sp)
		common.CheckErr(t, errRange, false, "range_filter must be greater than radius for IP/COSINE")
	}
}

// test range search with scann index and entity.HAMMING, entity.JACCARD, entity.TANIMOTO metric type
func TestRangeSearchScannBinary(t *testing.T) {
	// not supported  entity.TANIMOTO ??
	t.Parallel()
	for _, metricType := range []entity.MetricType{entity.HAMMING, entity.JACCARD} {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64BinaryVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64BinaryVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
		_, _ = insertData(ctx, t, mc, dp)
		mc.Flush(ctx, collName, false)

		// create scann index
		indexBin, _ := entity.NewIndexBinIvfFlat(metricType, 16)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, indexBin, false)
		common.CheckErr(t, err, true)

		// describe index
		indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultBinaryVecFieldName)
		expIndex := entity.NewGenericIndex(common.DefaultIndexName, entity.BinIvfFlat, indexBin.Params())
		common.CheckIndexResult(t, indexes, expIndex)

		// load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// range search filter distance and output all fields
		queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeBinaryVector)
		sp, _ := entity.NewIndexBinIvfFlatSearchParam(8)
		sp.AddRadius(100)
		sp.AddRangeFilter(0)
		resSearch, errSearch := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultBinaryVecFieldName,
			metricType, common.DefaultTopK, sp)

		// verify error nil, output all fields, range score
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, resSearch, 1, common.DefaultTopK)
		common.CheckOutputFields(t, resSearch[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName,
			common.DefaultBinaryVecFieldName, common.DefaultDynamicFieldName})
		for _, s := range resSearch[0].Scores {
			require.GreaterOrEqual(t, s, float32(0))
			require.Less(t, s, float32(100))
		}

		// invalid range search: radius > range filter
		sp.AddRadius(0)
		sp.AddRangeFilter(100)
		_, errRange := mc.Search(ctx, collName, []string{}, "", []string{"*"}, queryVec, common.DefaultBinaryVecFieldName,
			metricType, common.DefaultTopK, sp)
		common.CheckErr(t, errRange, false, "range_filter must be less than radius for L2/HAMMING/JACCARD")
	}
}

// TODO offset and limit
// TODO consistency level
// TODO WithGuaranteeTimestamp
// TODO ignore growing
