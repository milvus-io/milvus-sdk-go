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

	allFloatIndex := []entity.Index{
		idxFlat,
		idxIvfFlat,
		idxHnsw,
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
	idxIvfSq8, _ := entity.NewIndexIvfSQ8(entity.L2, 128)
	idxIvfPq, _ := entity.NewIndexIvfPQ(entity.L2, 128, 16, 8)
	idxScann, _ := entity.NewIndexSCANN(entity.L2, 16, false)
	idxDiskAnn, _ := entity.NewIndexDISKANN(entity.L2)
	return []entity.Index{
		idxIvfSq8,
		idxIvfPq,
		idxScann,
		idxDiskAnn,
	}
}

func prepareDataForGroupBySearch(t *testing.T, loopInsert int, idx entity.Index, withGrowing bool) (*base.MilvusClient, context.Context, string) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
	mc := createMilvusClient(ctx, t)

	// create collection with all datatype
	cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: 100, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	for i := 0; i < loopInsert; i++ {
		_, _ = insertData(ctx, t, mc, dp)
	}

	if !withGrowing {
		mc.Flush(ctx, collName, false)
	}

	// create vector index and scalar index
	supportedGroupByFields := []string{common.DefaultIntFieldName, "int8", "int16", "int32", "varchar", "bool"}
	idxScalar := entity.NewScalarIndex()
	for _, groupByField := range supportedGroupByFields {
		mc.CreateIndex(ctx, collName, groupByField, idxScalar, false)
		//common.CheckErr(t, err, true)
	}
	idxHnsw, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName != common.DefaultFloatVecFieldName {
			err := mc.CreateIndex(ctx, collName, fieldName, idxHnsw, false)
			common.CheckErr(t, err, true)
		}
	}
	err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// load collection
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	return mc, ctx, collName
}

// create coll with all datatype -> build all supported index
// -> search with WithGroupByField (int* + varchar + bool
// -> verify every top passage is the top of whole group
// output_fields: pk + groupBy
func TestSearchGroupByFloatDefault(t *testing.T) {
	t.Skip("unstable case and https://github.com/milvus-io/milvus/issues/31494")
	t.Parallel()
	for _, metricType := range common.SupportFloatMetricType {
		for _, idx := range genGroupByVectorIndex(metricType) {
			// prepare data
			mc, ctx, collName := prepareDataForGroupBySearch(t, 100, idx, false)

			// search params
			queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
			sp, _ := entity.NewIndexIvfFlatSearchParam(32)

			// search with groupBy field
			supportedGroupByFields := []string{common.DefaultIntFieldName, "int8", "int16", "int32", "varchar", "bool"}
			for _, groupByField := range supportedGroupByFields {
				resGroupBy, _ := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, groupByField}, queryVec,
					common.DefaultFloatVecFieldName, metricType, common.DefaultTopK, sp, client.WithGroupByField(groupByField))

				// verify each topK entity is the top1 of the whole group
				hitsNum := 0
				total := 0
				for _, rs := range resGroupBy {
					for i := 0; i < rs.ResultCount; i++ {
						groupByValue, _ := rs.Fields.GetColumn(groupByField).Get(i)
						pkValue, _ := rs.IDs.GetAsInt64(i)
						var expr string
						if groupByField == "varchar" {
							expr = fmt.Sprintf("%s == '%v' ", groupByField, groupByValue)
						} else {
							expr = fmt.Sprintf("%s == %v", groupByField, groupByValue)
						}

						// 	search filter with groupByValue is the top1
						resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{common.DefaultIntFieldName,
							groupByField}, queryVec, common.DefaultFloatVecFieldName, metricType, 1, sp)
						filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
						//log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
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
				//require.GreaterOrEqualf(t, hitsRate, float32(0.1), _str)
			}
		}
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
			cp := CollectionParams{CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: true,
				ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
			collName := createCollection(ctx, t, mc, cp)

			// insert
			dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: VarcharBinaryVec,
				start: 0, nb: 1000, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
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
				common.CheckErr(t, err, false, "Unsupported dataType for chunk brute force iterator:VECTOR_BINARY")
			}
		}
	}
}

// binary vector -> growing segments, maybe brute force
// default Bounded ConsistencyLevel -> succ ??
// strong ConsistencyLevel -> error
func TestSearchGroupByBinaryGrowing(t *testing.T) {
	//t.Skip("#31134")
	t.Parallel()
	for _, metricType := range common.SupportBinIvfFlatMetricType {
		idxBinIvfFlat, _ := entity.NewIndexBinIvfFlat(metricType, 128)
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		// connect
		mc := createMilvusClient(ctx, t)

		// create collection with all datatype
		cp := CollectionParams{CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// create index and load
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idxBinIvfFlat, false)
		common.CheckErr(t, err, true)
		err = mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, err, true)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: VarcharBinaryVec,
			start: 0, nb: 1000, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
		_, _ = insertData(ctx, t, mc, dp)

		// search params
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeBinaryVector)
		sp, _ := entity.NewIndexBinIvfFlatSearchParam(64)
		supportedGroupByFields := []string{common.DefaultVarcharFieldName}

		// search with groupBy field
		for _, groupByField := range supportedGroupByFields {
			_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultVarcharFieldName,
				groupByField}, queryVec, common.DefaultBinaryVecFieldName, metricType, common.DefaultTopK, sp,
				client.WithGroupByField(groupByField), client.WithSearchQueryConsistencyLevel(entity.ClStrong))
			common.CheckErr(t, err, false, "Unsupported dataType for chunk brute force iterator:VECTOR_BINARY")
		}
	}
}

// groupBy in growing segments, maybe growing index or brute force
func TestSearchGroupByFloatGrowing(t *testing.T) {
	for _, metricType := range common.SupportFloatMetricType {
		idxHnsw, _ := entity.NewIndexHNSW(metricType, 8, 96)
		mc, ctx, collName := prepareDataForGroupBySearch(t, 10, idxHnsw, true)

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
			for _, rs := range resGroupBy {
				for i := 0; i < rs.ResultCount; i++ {
					groupByValue, _ := rs.Fields.GetColumn(groupByField).Get(i)
					pkValue, _ := rs.IDs.GetAsInt64(i)
					var expr string
					if groupByField == "varchar" {
						expr = fmt.Sprintf("%s == '%v' ", groupByField, groupByValue)
					} else {
						expr = fmt.Sprintf("%s == %v", groupByField, groupByValue)
					}

					resFilter, _ := mc.Search(ctx, collName, []string{}, expr, []string{common.DefaultIntFieldName,
						groupByField}, queryVec, common.DefaultFloatVecFieldName, metricType, 1, sp)

					// search filter with groupByValue is the top1
					filterTop1Pk, _ := resFilter[0].IDs.GetAsInt64(0)
					//log.Printf("Search top1 with %s: groupByValue: %v, pkValue: %d. The returned pk by filter search is: %d",
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
			require.GreaterOrEqual(t, hitsRate, float32(0.6), _str)
		}
	}
}

// groupBy + pagination
func TestSearchGroupByPagination(t *testing.T) {
	// create index and load
	idx, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 100, idx, false)

	// search params
	queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	sp, _ := entity.NewIndexIvfFlatSearchParam(32)
	var offset = int64(10)

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
		mc, ctx, collName := prepareDataForGroupBySearch(t, 3, idx, false)

		// groupBy search
		queryVec := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		sp, _ := entity.NewIndexIvfFlatSearchParam(32)
		_, err := mc.Search(ctx, collName, []string{}, "", []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName},
			queryVec, common.DefaultFloatVecFieldName, entity.MetricType(idx.Params()["metrics_type"]),
			common.DefaultTopK, sp, client.WithGroupByField(common.DefaultVarcharFieldName))
		common.CheckErr(t, err, false, "trying to groupBy on unsupported index type will fail, "+
			"currently only support ivf-flat, ivf_cc and HNSW")
	}
}

// FLOAT, DOUBLE, JSON, ARRAY
func TestSearchGroupByUnsupportedDataType(t *testing.T) {
	idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 1, idxHnsw, true)

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
	mc, ctx, collName := prepareDataForGroupBySearch(t, 1, idxHnsw, true)

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

// groupBy + advanced search
func TestSearchGroupByHybridSearch(t *testing.T) {
	// prepare data
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc, ctx, collName := prepareDataForGroupBySearch(t, 100, indexHnsw, false)

	// hybrid search with groupBy field
	sp, _ := entity.NewIndexHNSWSearchParam(20)
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK, client.WithOffset(2)),
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK, client.WithGroupByField("varchar")),
	}
	//supportedGroupByFields := []string{common.DefaultIntFieldName, "int8", "int16", "int32", "varchar", "bool"}
	resGroupBy, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, resGroupBy, 1, common.DefaultTopK)
	log.Println(resGroupBy[0].IDs, resGroupBy[0].GroupByValue)
}
