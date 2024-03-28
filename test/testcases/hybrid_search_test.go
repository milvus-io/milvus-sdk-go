//go:build L0

package testcases

import (
	"fmt"
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
	ctx := createContext(t, time.Second*common.DefaultTimeout*2)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, enableDynamic := range []bool{false} {
		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

		dp := DataParams{DoInsert: true, CollectionFieldsType: AllFields, start: 0, nb: common.DefaultNb,
			dim: common.DefaultDim, EnableDynamicField: enableDynamic}

		// index params
		indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		ips := make([]IndexParams, 4)
		for _, fieldName := range common.AllVectorsFieldsName {
			ips = append(ips, IndexParams{BuildIndex: true, Index: indexHnsw, FieldName: fieldName, async: false})
		}

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
		queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
		queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)

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
				common.CheckSearchResult(t, searchRes, 1, limit.expLimit)
				common.CheckOutputFields(t, searchRes[0].Fields, common.GetAllFieldsName(enableDynamic))
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
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	ips := make([]IndexParams, 4)
	for _, fieldName := range common.AllVectorsFieldsName {
		ips = append(ips, IndexParams{BuildIndex: true, Index: indexHnsw, FieldName: fieldName, async: false})
	}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips),
		WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search with invalid limit
	ranker := client.NewRRFReranker()
	sp, _ := entity.NewIndexFlatSearchParam()
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeBinaryVector)
	sReqs := []*client.ANNSearchRequest{
		client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK),
		client.NewANNSearchRequest(common.DefaultBinaryVecFieldName, entity.L2, "", queryVec2, sp, common.DefaultTopK),
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
	queryVecNq := common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVecType := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)
	queryVecDim := common.GenSearchVectors(1, common.DefaultDim*2, entity.FieldTypeFloatVector)
	sReqs := [][]*client.ANNSearchRequest{
		{client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecNq, sp, common.DefaultTopK)},   // nq != 1
		{client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecType, sp, common.DefaultTopK)}, // TODO vector type not match
		{client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVecDim, sp, common.DefaultTopK)},  // vector dim not match
	}
	for _, invalidSReq := range sReqs {
		_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, ranker, invalidSReq)
		common.CheckErr(t, errSearch, false, "nq should be equal to 1", "vector dimension mismatch")
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

	dp := DataParams{DoInsert: true, CollectionFieldsType: AllVectors, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: false}

	// index params
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	ips := make([]IndexParams, 4)
	for _, fieldName := range common.AllVectorsFieldsName {
		ips = append(ips, IndexParams{BuildIndex: true, Index: indexHnsw, FieldName: fieldName, async: false})
	}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search with different limit
	sp, _ := entity.NewIndexFlatSearchParam()
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)
	for _, invalidOffset := range []int64{-1, common.MaxTopK + 1} {
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, "", queryVec1, sp, common.DefaultTopK, client.WithOffset(invalidOffset)),
			client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, "", queryVec2, sp, common.DefaultTopK),
		}
		_, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, client.NewRRFReranker(), sReqs)
		common.CheckErr(t, errSearch, false, "top k should be in range [1, 16384]")
	}

	// search with different reranker and offset
	for _, reranker := range []client.Reranker{
		client.NewRRFReranker(),
		client.NewWeightedReranker([]float64{0.8, 0.2}),
		client.NewWeightedReranker([]float64{0.0, 0.2}),
		client.NewWeightedReranker([]float64{0.4, 1.0}),
	} {
		sReqs := []*client.ANNSearchRequest{
			client.NewANNSearchRequest(common.DefaultFloatVecFieldName, entity.L2, expr, queryVec1, sp, common.DefaultTopK, client.WithOffset(5)),
			client.NewANNSearchRequest(common.DefaultFloat16VecFieldName, entity.L2, expr, queryVec2, sp, common.DefaultTopK),
		}
		// hybrid search
		searchRes, errSearch := mc.HybridSearch(ctx, collName, []string{}, common.DefaultTopK, []string{}, reranker, sReqs)
		common.CheckErr(t, errSearch, true)
		common.CheckSearchResult(t, searchRes, 1, common.DefaultTopK)
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
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	ips := make([]IndexParams, 4)
	for _, fieldName := range common.AllVectorsFieldsName {
		ips = append(ips, IndexParams{BuildIndex: true, Index: indexHnsw, FieldName: fieldName, async: false})
	}

	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithIndexParams(ips), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// hybrid search
	sp, _ := entity.NewIndexFlatSearchParam()
	expr := fmt.Sprintf("%s > 4", common.DefaultIntFieldName)
	queryVec1 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
	queryVec2 := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloat16Vector)

	// search with different reranker and offset
	sp.AddRadius(5)
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
				require.LessOrEqual(t, score, float32(5))
			}
		}
	}
}
