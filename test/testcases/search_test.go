//go:build L0

package testcases

import (
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestSearch(t *testing.T) {
	t.Skip("Skip for index option and index return")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, false, true)

	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true, "")
	segments, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	for _, segment := range segments {
		log.Printf("segment id: %v, num rows: %v", segment.ID, segment.NumRows)
	}

	// prepare search vec
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	searchResult, _ := mc.Search(
		ctx,                                    // ctx
		collName,                               // CollectionName
		[]string{common.DefaultPartition},      // partitionNames
		"",                                     // expr
		[]string{common.DefaultFloatFieldName}, // outputFields
		//[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim), // vectors
		common.DefaultFloatVecFieldName,                              // vectorField
		entity.L2,                                                    // metricType
		common.DefaultTopK,                                           // topK
		sp,                                                           // sp
	)

	common.CheckOutputFields(t, searchResult[0].Fields, []string{common.DefaultFloatFieldName})
	common.CheckSearchResult(t, searchResult, common.DefaultNq, common.DefaultTopK)

	for _, result := range searchResult {
		log.Printf("result count: %d, score: %v", result.ResultCount, result.Scores)
		for _, column := range result.Fields {
			log.Printf("search retun field name is %v", column.Name())
			log.Printf("return int fields data: %v", column.(*entity.ColumnFloat).Data())
		}
		log.Printf("result score %v", result.Scores)
		if result.IDs.Type() == entity.FieldTypeInt64 {
			ids := result.IDs.(*entity.ColumnInt64).Data()
			log.Printf("search ids is %v", ids)
		}
	}
}
