//go:build L0

package testcases

import (
	"context"
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/test/base"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"

	"github.com/stretchr/testify/require"
)

// Generate groupBy-supported vector indexes
func genGroupByVectorIndex(metricType entity.MetricType) []entity.Index {
	nlist := 128
	idxFlat, _ := entity.NewIndexFlat(metricType)
	idxIvfFlat, _ := entity.NewIndexIvfFlat(metricType, nlist)
	idxHnsw, _ := entity.NewIndexHNSW(metricType, 8, 96)
	idxIvfSq8, _ := entity.NewIndexIvfSQ8(metricType, 128)

	allFloatIndex := []entity.Index{
		idxFlat,
		idxIvfFlat,
		idxHnsw,
		idxIvfSq8,
	}
	return allFloatIndex
}

// Generate groupBy-supported vector indexes
func genGroupByBinaryIndex(metricType entity.MetricType) []entity.Index {
	nlist := 128
	idxBinFlat, _ := entity.NewIndexBinFlat(metricType, nlist)
	idxBinIvfFlat, _ := entity.NewIndexBinIvfFlat(metricType, nlist)

	allFloatIndex := []entity.Index{
		idxBinFlat,
		idxBinIvfFlat,
	}
	return allFloatIndex
}

func genUnsupportedFloatGroupByIndex() []entity.Index {
	idxIvfPq, _ := entity.NewIndexIvfPQ(entity.L2, 128, 16, 8)
	idxScann, _ := entity.NewIndexSCANN(entity.L2, 16, false)
	return []entity.Index{
		idxIvfPq,
		idxScann,
	}
}

func prepareDataForGroupBySearch(t *testing.T, loopInsert int, insertNi int, idx entity.Index, withGrowing bool) (*base.MilvusClient, context.Context, string) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
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
		start: 0, nb: insertNi, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	for i := 0; i < loopInsert; i++ {
		_, _ = insertData(ctx, t, mc, dp)
	}

	if !withGrowing {
		mc.Flush(ctx, collName, false)
	}

	// create scalar index
	supportedGroupByFields := []string{
		common.DefaultIntFieldName, common.DefaultInt8FieldName, common.DefaultInt16FieldName,
		common.DefaultInt32FieldName, common.DefaultVarcharFieldName, common.DefaultBoolFieldName,
	}
	for _, groupByField := range supportedGroupByFields {
		err := mc.CreateIndex(ctx, collName, groupByField, entity.NewScalarIndex(), false)
		common.CheckErr(t, err, true)
	}

	// create vector index
	idxHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultFloatVecFieldName {
			err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
			common.CheckErr(t, err, true)
		} else if fieldName == common.DefaultBinaryVecFieldName {
			err := mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
			common.CheckErr(t, err, true)
		} else {
			err := mc.CreateIndex(ctx, collName, fieldName, idxHnsw, false)
			common.CheckErr(t, err, true)
		}
	}

	// load collection
	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	return mc, ctx, collName
}

// create coll with all datatype -> build all supported index
// -> search with WithGroupByField (int* + varchar + bool
// -> verify every top passage is the top of whole group
// output_fields: pk + groupBy
func TestSearchGroupByFloatDefault(t *testing.T) {
	t.Skip("timeout case")
	t.Parallel()
	for _, metricType := range common.SupportFloatMetricType {
		for _, idx := range genGroupByVectorIndex(metricType) {
			// prepare data
			mc, ctx, collName := prepareDataForGroupBySearch(t, 100, 200, idx, false)

			// search params
			queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
			sp, _ := entity.NewIndexIvfFlatSearchParam(32)

			collection, _ := mc.DescribeCollection(ctx, collName)
			common.PrintAllFieldNames(collName, collection.Schema)

			// search with groupBy field
			supportedGroupByFields := []string{common.DefaultIntFieldName, "int8", "int16", "int32", "varchar", "bool"}
			for _, groupByField := range supportedGroupByFields {
				resGroupBy, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, groupByField}, queryVec,
					common.DefaultFloatVecFieldName, metricType, common.DefaultTopK, sp, client.WithGroupByField(groupByField))

				// verify each topK entity is the top1 of the whole group
				hitsNum := 0
				total := 0
				for i := 0; i < common.DefaultNq; i++ {
					for j := 0; j < resGroupBy[i].ResultCount; j++ {
						groupByValue, _ := resGroupBy[i].GroupByValue.Get(j)
						pkValue, _ := resGroupBy[i].IDs.GetAsInt64(j)
						var expr string
						if groupByField == "varchar" {
							expr = fmt.Sprintf("%s == '%v' ", groupByField, groupByValue)
						} else {
							expr = fmt.Sprintf("%s == %v", groupByField, groupByValue)
						}
						// 	search filter with groupByValue is the top1
						resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{
							common.DefaultIntFieldName,
							groupByField,
						}, []entity.Vector{queryVec[i]}, common.DefaultFloatVecFieldName, metricType, 1, sp)
						filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
						// log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
						//	groupByField, groupByValue, pkValue, filterTop1Pk)
						if filterTop1Pk == pkValue {
							hitsNum += 1
						}
						total += 1
					}
				}

				// verify hits rate
				hitsRate := float32(hitsNum) / float32(total)
				_str := fmt.Sprintf("GroupBy search with field %s, nq=%d and limit=%d , then hitsNum= %d, hitsRate=%v\n",
					groupByField, common.DefaultNq, common.DefaultTopK, hitsNum, hitsRate)
				log.Println(_str)
				if groupByField != "bool" {
					// waiting for fix https://github.com/milvus-io/milvus/issues/32630
					require.GreaterOrEqualf(t, hitsRate, float32(0.1), _str)
				}
			}
		}
	}
}

// test groupBy search sparse vector
func TestGroupBySearchSparseVector(t *testing.T) {
	t.Parallel()
	idxInverted, _ := entity.NewIndexSparseInverted(entity.IP, 0.3)
	idxWand, _ := entity.NewIndexSparseWAND(entity.IP, 0.2)
	for _, idx := range []entity.Index{idxInverted, idxWand} {
		ctx := createContext(t, time.Second*common.DefaultTimeout*2)
		// connect
		mc := createMilvusClient(ctx, t)

		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen,
		}
		collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

		// insert data
		dp := DataParams{
			DoInsert: true, CollectionName: collName, CollectionFieldsType: Int64VarcharSparseVec, start: 0,
			nb: 200, dim: common.DefaultDim, EnableDynamicField: true,
		}
		for i := 0; i < 100; i++ {
			_, _ = insertData(ctx, t, mc, dp)
		}
		mc.Flush(ctx, collName, false)

		// index and load
		idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false)
		mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)
		mc.LoadCollection(ctx, collName, false)

		// groupBy search
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeSparseVector)
		sp, _ := entity.NewIndexSparseInvertedSearchParam(0.2)
		resGroupBy, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName}, queryVec,
			common.DefaultSparseVecFieldName, entity.IP, common.DefaultTopK, sp, client.WithGroupByField(common.DefaultVarcharFieldName))

		// verify each topK entity is the top1 of the whole group
		hitsNum := 0
		total := 0
		for i := 0; i < common.DefaultNq; i++ {
			if resGroupBy[i].ResultCount > 0 {
				for j := 0; j < resGroupBy[i].ResultCount; j++ {
					groupByValue, _ := resGroupBy[i].GroupByValue.Get(j)
					pkValue, _ := resGroupBy[i].IDs.GetAsInt64(j)
					expr := fmt.Sprintf("%s == '%v' ", common.DefaultVarcharFieldName, groupByValue)
					// 	search filter with groupByValue is the top1
					resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{
						common.DefaultIntFieldName,
						common.DefaultVarcharFieldName,
					}, []entity.Vector{queryVec[i]}, common.DefaultSparseVecFieldName, entity.IP, 1, sp)
					filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
					log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
						common.DefaultVarcharFieldName, groupByValue, pkValue, filterTop1Pk)
					if filterTop1Pk == pkValue {
						hitsNum += 1
					}
					total += 1
				}
			}
		}

		// verify hits rate
		hitsRate := float32(hitsNum) / float32(total)
		_str := fmt.Sprintf("GroupBy search with field %s, nq=%d and limit=%d , then hitsNum= %d, hitsRate=%v\n",
			common.DefaultVarcharFieldName, common.DefaultNq, common.DefaultTopK, hitsNum, hitsRate)
		log.Println(_str)
		require.GreaterOrEqualf(t, hitsRate, float32(0.8), _str)
	}
}

// binary vector -> not supported
func TestSearchGroupByBinaryDefault(t *testing.T) {
	t.Parallel()
	for _, metricType := range common.SupportBinIvfFlatMetricType {
		for _, idx := range genGroupByBinaryIndex(metricType) {
			ctx := createContext(t, time.Second*common.DefaultTimeout)
			// connect
			mc := createMilvusClient(ctx, t)

			// create collection with all datatype
			cp := CollectionParams{
				CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: true,
				ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
			}
			collName := createCollection(ctx, t, mc, cp)

			// insert
			dp := DataParams{
				CollectionName: collName, PartitionName: "", CollectionFieldsType: VarcharBinaryVec,
				start: 0, nb: 1000, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
			}
			for i := 0; i < 2; i++ {
				_, _ = insertData(ctx, t, mc, dp)
			}
			mc.Flush(ctx, collName, false)

			// create index and load
			err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false)
			common.CheckErr(t, err, true)
			err = mc.LoadCollection(ctx, collName, false)
			common.CheckErr(t, err, true)

			// search params
			queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeBinaryVector)
			sp, _ := entity.NewIndexBinIvfFlatSearchParam(32)
			supportedGroupByFields := []string{common.DefaultVarcharFieldName, common.DefaultBinaryVecFieldName}

			// search with groupBy field
			for _, groupByField := range supportedGroupByFields {
				_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultVarcharFieldName, groupByField}, queryVec,
					common.DefaultBinaryVecFieldName, metricType, common.DefaultTopK, sp, client.WithGroupByField(groupByField))
				common.CheckErr(t, err, false, "not support search_group_by operation based on binary vector column")
			}
		}
	}
}

// binary vector -> growing segments, maybe brute force
// default Bounded ConsistencyLevel -> succ ??
// strong ConsistencyLevel -> error
func TestSearchGroupByBinaryGrowing(t *testing.T) {
	t.Parallel()
	for _, metricType := range common.SupportBinIvfFlatMetricType {
		idxBinIvfFlat, _ := entity.NewIndexBinIvfFlat(metricType, 128)
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection with all datatype
		cp := CollectionParams{
			CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp)

		// create index and load
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinIvfFlat, false)
		common.CheckErr(t, err, true)
		err = mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, err, true)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: VarcharBinaryVec,
			start: 0, nb: 1000, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
		}
		_, _ = insertData(ctx, t, mc, dp)

		// search params
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
		sp, _ := entity.NewIndexBinIvfFlatSearchParam(64)
		supportedGroupByFields := []string{common.DefaultVarcharFieldName}

		// search with groupBy field
		for _, groupByField := range supportedGroupByFields {
			_, err := mc.Search(ctx, collName, []string{}, "", []string{
				common.DefaultVarcharFieldName,
				groupByField,
			}, queryVec, common.DefaultBinaryVecFieldName, metricType, common.DefaultTopK, sp,
				client.WithGroupByField(groupByField), client.WithSearchQueryConsistencyLevel(entity.ClStrong))
			common.CheckErr(t, err, false, "not support search_group_by operation based on binary vector column")
		}
	}
}

// groupBy in growing segments, maybe growing index or brute force
func TestSearchGroupByFloatGrowing(t *testing.T) {
	for _, metricType := range common.SupportFloatMetricType {
		idxHnsw, _ := entity.NewIndexHNSW(metricType, 8, 96)
		mc, ctx, collName := prepareDataForGroupBySearch(t, 100, 200, idxHnsw, true)

		// search params
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexIvfFlatSearchParam(32)
		supportedGroupByFields := []string{common.DefaultIntFieldName, "int8", "int16", "int32", "varchar", "bool"}

		// search with groupBy field
		hitsNum := 0
		total := 0
		for _, groupByField := range supportedGroupByFields {
			resGroupBy, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, groupByField}, queryVec,
				common.DefaultFloatVecFieldName, metricType, common.DefaultTopK, sp, client.WithGroupByField(groupByField),
				client.WithSearchQueryConsistencyLevel(entity.ClStrong))

			// verify each topK entity is the top1 in the group
			for i := 0; i < common.DefaultNq; i++ {
				for j := 0; j < resGroupBy[i].ResultCount; j++ {
					groupByValue, _ := resGroupBy[i].GroupByValue.Get(j)
					pkValue, _ := resGroupBy[i].IDs.GetAsInt64(j)
					var expr string
					if groupByField == "varchar" {
						expr = fmt.Sprintf("%s == '%v' ", groupByField, groupByValue)
					} else {
						expr = fmt.Sprintf("%s == %v", groupByField, groupByValue)
					}
					resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{
						common.DefaultIntFieldName,
						groupByField,
					}, []entity.Vector{queryVec[i]}, common.DefaultFloatVecFieldName, metricType, 1, sp, client.WithSearchQueryConsistencyLevel(entity.ClStrong))

					// search filter with groupByValue is the top1
					filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
					// log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
					//	groupByField, groupByValue, pkValue, filterTop1Pk)
					if filterTop1Pk == pkValue {
						hitsNum += 1
					}
					total += 1
				}
			}
			// verify hits rate
			hitsRate := float32(hitsNum) / float32(total)
			_str := fmt.Sprintf("GroupBy search with field %s, nq=%d and limit=%d , then hitsNum= %d, hitsRate=%v\n",
				groupByField, common.DefaultNq, common.DefaultTopK, hitsNum, hitsRate)
			log.Println(_str)
			if groupByField != "bool" {
				require.GreaterOrEqualf(t, hitsRate, float32(0.8), _str)
			}
		}
	}
}

// groupBy + pagination
func TestSearchGroupByPagination(t *testing.T) {
	// create index and load
	idx, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 10, 1000, idx, false)

	// search params
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexIvfFlatSearchParam(32)
	offset := int64(10)

	// search pagination & groupBy
	resGroupByPagination, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
		queryVec, common.DefaultFloatVecFieldName, entity.COSINE, common.DefaultTopK, sp,
		client.WithGroupByField(common.DefaultVarcharFieldName), client.WithOffset(offset))

	common.CheckSearchResult(t, resGroupByPagination, common.DefaultNq, common.DefaultTopK)

	// search limit=origin limit + offset
	resGroupByDefault, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
		queryVec, common.DefaultFloatVecFieldName, entity.COSINE, common.DefaultTopK+int(offset), sp,
		client.WithGroupByField(common.DefaultVarcharFieldName))
	for i := 0; i < common.DefaultNq; i++ {
		require.Equal(t, resGroupByDefault[i].IDs.(*entity.ColumnInt64).Data()[10:], resGroupByPagination[i].IDs.(*entity.ColumnInt64).Data())
	}
}

// only support: "FLAT", "IVF_FLAT", "HNSW"
func TestSearchGroupByUnsupportedIndex(t *testing.T) {
	t.Parallel()
	for _, idx := range genUnsupportedFloatGroupByIndex() {
		mc, ctx, collName := prepareDataForGroupBySearch(t, 3, 1000, idx, false)
		// groupBy search
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexIvfFlatSearchParam(32)
		_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
			queryVec, common.DefaultFloatVecFieldName, entity.MetricType(idx.Params()["metrics_type"]),
			common.DefaultTopK, sp, client.WithGroupByField(common.DefaultVarcharFieldName))
		common.CheckErr(t, err, false, "doesn't support")
	}
}

// FLOAT, DOUBLE, JSON, ARRAY
func TestSearchGroupByUnsupportedDataType(t *testing.T) {
	idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 1, 1000, idxHnsw, true)

	// groupBy search with unsupported field type
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexIvfFlatSearchParam(32)
	for _, unsupportedField := range []string{"float", "double", "json", "floatVec", "int8Array", "floatArray"} {
		_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
			queryVec, common.DefaultFloatVecFieldName, entity.L2,
			common.DefaultTopK, sp, client.WithGroupByField(unsupportedField))
		common.CheckErr(t, err, false, "unsupported data type")
	}
}

// groupBy + iterator -> not supported
func TestSearchGroupByIterator(t *testing.T) {
	// TODO: sdk support
}

// groupBy + range search -> not supported
func TestSearchGroupByRangeSearch(t *testing.T) {
	idxHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 1, 1000, idxHnsw, true)

	// groupBy search with range
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(50)
	sp.AddRadius(0)
	sp.AddRangeFilter(0.8)

	// range search
	_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
		queryVec, common.DefaultFloatVecFieldName, entity.COSINE, common.DefaultTopK, sp,
		client.WithGroupByField(common.DefaultVarcharFieldName))

	common.CheckErr(t, err, false, "Not allowed to do range-search when doing search-group-by")
}

func TestSearchGroupByHybridSearch(t *testing.T) {
	t.Skip("case panic")
	indexHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 10, 1000, indexHnsw, false)

	// search params
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexHNSWSearchParam(20)

	collection, _ := mc.DescribeCollection(ctx, collName)
	common.PrintAllFieldNames(collName, collection.Schema)

	// search with groupBy field
	groupByField := common.DefaultVarcharFieldName

	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.COSINE, expr, queryVec1, sp, common.DefaultTopK, client.WithOffset(2)),
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.COSINE, expr, queryVec2, sp, common.DefaultTopK),
	}
	resGroupBy, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs, client.WithGroupByField(groupByField))
	common.CheckErr(t, errSearch, true)

	// verify each topK entity is the top1 of the whole group
	hitsNum := 0
	total := 0
	for i := 0; i < common.DefaultNq; i++ {
		for j := 0; j < resGroupBy[i].ResultCount; j++ {
			groupByValue, _ := resGroupBy[i].GroupByValue.Get(j)
			pkValue, _ := resGroupBy[i].IDs.GetAsInt64(j)
			expr = fmt.Sprintf("%s == '%v' ", groupByField, groupByValue)
			// 	search filter with groupByValue is the top1
			resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{
				common.DefaultIntFieldName,
				groupByField,
			}, []entity.Vector{queryVec[i]}, common.DefaultFloatVecFieldName, entity.COSINE, 1, sp)
			filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
			// log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
			//	groupByField, groupByValue, pkValue, filterTop1Pk)
			if filterTop1Pk == pkValue {
				hitsNum += 1
			}
			total += 1
		}
	}

	// verify hits rate
	hitsRate := float32(hitsNum) / float32(total)
	_str := fmt.Sprintf("GroupBy search with field %s, nq=%d and limit=%d , then hitsNum= %d, hitsRate=%v\n",
		groupByField, common.DefaultNq, common.DefaultTopK, hitsNum, hitsRate)
	log.Println(_str)
	require.GreaterOrEqualf(t, hitsRate, float32(0.1), _str)
}

// groupBy + advanced search
func TestHybridSearchDifferentGroupByField(t *testing.T) {
	t.Skip("TODO: 2.5 test hybrid search with groupBy")
	// prepare data
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 5, 1000, indexHnsw, false)

	// hybrid search with groupBy field
	sp, _ := entity.NewIndexHNSWSearchParam(20)
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(2, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(2, common.DefaultDim, entity.FieldTypeFloatVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK, client.WithOffset(2), client.WithGroupByField("int64")),
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK, client.WithGroupByField("varchar")),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs, client.WithGroupByField("int8"))
	common.CheckErr(t, errSearch, true)
	// TODO check the true groupBy field
}

// groupBy field not existed
func TestSearchNotExistedGroupByField(t *testing.T) {
	t.Skip("https://github.com/milvus-io/milvus-sdk-go/issues/828")
	// prepare data
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 2, 1000, indexHnsw, false)

	// hybrid search with groupBy field
	sp, _ := entity.NewIndexHNSWSearchParam(20)
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(2, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(2, common.DefaultDim, entity.FieldTypeFloatVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK, client.WithOffset(2), client.WithGroupByField("aaa")),
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK, client.WithGroupByField("bbb")),
	}
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs, client.WithGroupByField("ccc"))
	common.CheckErr(t, errSearch, false, "groupBy field not found in schema: field not found[field=ccc]")

	// search
	_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
		queryVec1, common.DefaultFloatVecFieldName, entity.L2, common.DefaultTopK, sp, client.WithGroupByField("ddd"))
	common.CheckErr(t, err, false, "groupBy field not found in schema: field not found[field=ddd]")
}
