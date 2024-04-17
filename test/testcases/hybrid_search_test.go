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

func TestHybridSearchDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: Int64FloatVec, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search
	ranker := client.NewRRFReranker()
	expr := fmt.Sprintf("%s > 10", common.DefaultIntFieldName)
	sp, _ := entity.NewIndexFlatSearchParam()
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK),
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK),
	}
	searchRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{"*"}, ranker, sReqs)
	common.CheckErr(t, errSearch, true)
	common.CheckSearchResult(t, searchRes, 1, common.DefaultTopK)
	common.CheckOutputFields(t, searchRes[0].Fields, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName})
}

// hybrid search default -> verify success
func TestHybridSearchMultiVectorsDefault(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, enableDynamic := range []bool{false, true} {
		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

		dp := DataParams{DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb * 3,
			dim: common.DefaultDim, EnableDynamicField: enableDynamic}

		ips := GenDefaultIndexParamsForAllVectors()

		collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

		// hybrid search with different limit
		type limitGroup struct {
			limit1   int
			limit2   int
			limit3   int
			expLimit int
		}
		limits := []limitGroup{
			{limit1: 10, limit2: 5, limit3: 8, expLimit: 8},
			{limit1: 10, limit2: 5, limit3: 15, expLimit: 15},
			{limit1: 10, limit2: 5, limit3: 20, expLimit: 15},
		}
		sp, _ := entity.NewIndexFlatSearchParam()
		expr := fmt.Sprintf("%s > 5", common.DefaultIntFieldName)
		queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloat16Vector)

		// search with different reranker and limit
		for _, reranker := range []client.Reranker{client.NewRRFReranker(),
			client.NewWeightedReranker([]float64{0.8, 0.2}),
			client.NewWeightedReranker([]float64{0.0, 0.2}),
			client.NewWeightedReranker([]float64{0.4, 1.0}),
		} {
			for _, limit := range limits {
				// hybrid search
				sReqs := []*client.ANNSearchRequest{
					client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, limit.limit1),
					client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, expr, queryVec2, sp, limit.limit2),
				}
				searchRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, limit.limit3, []string{"*"}, reranker, sReqs)
				common.CheckErr(t, errSearch, true)
				common.CheckSearchResult(t, searchRes, common.DefaultNq, limit.expLimit)
				common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(enableDynamic, false))
			}
		}
	}
}

// invalid limit: 0, -1, max+1
// invalid WeightedReranker params
// invalid fieldName: not exist
// invalid metric type: mismatch
func TestHybridSearchInvalidParams(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: AllVectors, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false}

	// index params
	ips := GenDefaultIndexParamsForAllVectors()
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search with invalid limit
	ranker := client.NewRRFReranker()
	sp, _ := entity.NewIndexFlatSearchParam()
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	for _, invalidLimit := range []int{-1, 0, common.MaxTopK + 1} {
		sReqsInvalid := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, invalidLimit)}

		for _, sReq := range [][]*client.ANNSearchRequest{sReqs, sReqsInvalid} {
			_, errSearch := mc.HybridSearch(ctx, collName, []string{}, invalidLimit, []string{}, ranker, sReq)
			common.CheckErr(t, errSearch, false, "should be greater than 0", "should be in range [1, 16384]")
		}
	}

	// hybrid search with invalid WeightedReranker params
	for _, invalidRanker := range []client.Reranker{
		client.NewWeightedReranker([]float64{-1, 0.2}),
		client.NewWeightedReranker([]float64{1.2, 0.2}),
		client.NewWeightedReranker([]float64{0.2}),
		client.NewWeightedReranker([]float64{0.2, 0.7, 0.5}),
	} {
		_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, invalidRanker, sReqs)
		common.CheckErr(t, errSearch, false, "rank param weight should be in range [0, 1]",
			"the length of weights param mismatch with ann search requests")
	}

	// invalid fieldName: not exist
	sReqs = append(sReqs, client.NewANNSearchRequest("a", entity.L2, "", queryVec1, sp, common.DefaultTopK))
	_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, ranker, sReqs)
	common.CheckErr(t, errSearch, false, "failed to get field schema by name: fieldName(a) not found")

	// invalid metric type: mismatch
	sReqsInvalidMetric := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.COSINE, "", queryVec1, sp, common.DefaultTopK),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVec2, sp, common.DefaultTopK),
	}
	_, errSearch = mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, ranker, sReqsInvalidMetric)
	common.CheckErr(t, errSearch, false, "metric type not match: invalid parameter")
}

// len(nq) != 1
// vector type mismatch: vectors: float32, queryVec: binary
// vector dim mismatch
func TestHybridSearchInvalidVectors(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: Int64FloatVec, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search with invalid limit
	ranker := client.NewRRFReranker()
	sp, _ := entity.NewIndexFlatSearchParam()
	// queryVecNq := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVecBinary := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeBinaryVector)
	queryVecType := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)
	queryVecDim := common.GenSearchVectors(1, common.DefaultDim*2, entity.FieldTypeFloatVector)
	sReqs := [][]*client.ANNSearchRequest{
		// {client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecNq, sp, common.DefaultTopK)},           // nq != 1
		{client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecType, sp, common.DefaultTopK)},         // TODO vector type not match
		{client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecDim, sp, common.DefaultTopK)},          // vector dim not match
		{client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.JACCARD, "", queryVecBinary, sp, common.DefaultTopK)}, // not exist vector types
	}
	for idx, invalidSReq := range sReqs {
		log.Println(idx)
		_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, ranker, invalidSReq)
		common.CheckErr(t, errSearch, false, "nq should be equal to 1", "vector dimension mismatch",
			"failed to get field schema by name", "vector type must be the same")
	}
}

// hybrid search Pagination -> verify success
func TestHybridSearchMultiVectorsPagination(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: AllVectors, start: 0, nb: common.DefaultNb * 5,
		dim: common.DefaultDim, EnableDynamicField: false}

	// index params
	ips := GenDefaultIndexParamsForAllVectors()
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search with different limit
	sp, _ := entity.NewIndexFlatSearchParam()
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)
	// milvus ignore invalid offset with ANNSearchRequest
	for _, invalidOffset := range []int64{-1, common.MaxTopK + 1} {
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(invalidOffset)),
			client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, "", queryVec2, sp, common.DefaultTopK),
		}
		_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
		common.CheckErr(t, errSearch, true)

		//hybrid search with invalid offset
		_, errSearch = mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs, client.WithOffset(invalidOffset))
		common.CheckErr(t, errSearch, false, "should be gte than 0", "(offset+limit) should be in range [1, 16384]")
	}

	// search with different reranker and offset
	for _, reranker := range []client.Reranker{
		client.NewRRFReranker(),
		client.NewWeightedReranker([]float64{0.8, 0.2}),
		client.NewWeightedReranker([]float64{0.0, 0.2}),
		client.NewWeightedReranker([]float64{0.4, 1.0}),
	} {
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK),
			client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK),
		}
		// hybrid search
		searchRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, reranker, sReqs)
		common.CheckErr(t, errSearch, true)
		offsetRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, 5, []string{}, reranker, sReqs, client.WithOffset(5))
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, searchRes, 1, common.DefaultTopK)
		common.CheckSearchResult(t, offsetRes, 1, 5)
		for i := 0; i < len(searchRes); i++ {
			require.Equal(t, searchRes[i].IDs.(*entity.ColumnInt64).Data()[5:], offsetRes[i].IDs.(*entity.ColumnInt64).Data())
		}
	}
}

// hybrid search Pagination -> verify success
func TestHybridSearchMultiVectorsRangeSearch(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout*5)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: AllVectors, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: AllVectors, start: 0, nb: common.DefaultNb * 3,
		dim: common.DefaultDim, EnableDynamicField: false}

	// index params
	ips := GenDefaultIndexParamsForAllVectors()
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search
	sp, _ := entity.NewIndexFlatSearchParam()
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)

	// search with different reranker and offset
	sp.AddRadius(20)
	sp.AddRangeFilter(0.01)
	for _, reranker := range []client.Reranker{
		client.NewRRFReranker(),
		client.NewWeightedReranker([]float64{0.8, 0.2}),
		client.NewWeightedReranker([]float64{0.5, 0.5}),
	} {
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK*2, client.WithOffset(1)),
			client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK),
		}
		// hybrid search
		resRange, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, reranker, sReqs)
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, resRange, 1, common.DefaultTopK)
		for _, res := range resRange {
			for _, score := range res.Scores {
				require.GreaterOrEqual(t, score, float32(0.01))
				require.LessOrEqual(t, score, float32(20))
			}
		}
	}
}

func TestHybridSearchSparseVector(t *testing.T) {
	t.Parallel()
	idxInverted := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_INVERTED_INDEX", map[string]string{"drop_ratio_build": "0.2", "metric_type": "IP"})
	idxWand := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_WAND", map[string]string{"drop_ratio_build": "0.3", "metric_type": "IP"})
	for _, idx := range []entity.Index{idxInverted, idxWand} {
		ctx := createContext(t, time.Second*common.DefaultTimeout*2)
		// connect
		mc := createMilvusClient(ctx, t)

		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen}

		dp := DataParams{DoInsert: true, CollectionFieldsType: Int64VarcharSparseVec, start: 0, nb: common.DefaultNb * 3,
			dim: common.DefaultDim, EnableDynamicField: true}

		// index params
		idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		ips := []IndexParams{
			{BuildIndex: true, Index: idx, FieldName: common.DefaultSparseVecFieldName, async: false},
			{BuildIndex: true, Index: idxHnsw, FieldName: common.DefaultFloatVecFieldName, async: false},
		}
		collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

		// search
		queryVec1 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim*2, entity.FieldTypeSparseVector)
		queryVec2 := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
		sp1, _ := entity.NewIndexSparseInvertedSearchParam(0.2)
		sp2, _ := entity.NewIndexHNSWSearchParam(20)
		expr := fmt.Sprintf("%s > 1", common.DefaultIntFieldName)
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultSparseVecFieldName, entity.IP, expr, queryVec1, sp1, common.DefaultTopK),
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec2, sp2, common.DefaultTopK),
		}
		for _, reranker := range []client.Reranker{
			client.NewRRFReranker(),
			client.NewWeightedReranker([]float64{0.5, 0.6}),
		} {
			// hybrid search
			searchRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{"*"}, reranker, sReqs)
			common.CheckErr(t, errSearch, true)
			common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
			common.CheckErr(t, errSearch, true)
			outputFields := []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName, common.DefaultFloatVecFieldName,
				common.DefaultSparseVecFieldName, common.DefaultDynamicFieldName}
			common.CheckOutputFields(t, searchRes[0].Fields, outputFields)
		}
	}
}
