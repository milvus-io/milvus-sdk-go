package testcases

import (
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestQuery(t *testing.T) {
	t.Skip("Skip for index option and index return")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection with data
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, false, false)

	// TODO create index

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, true)
	common.CheckErr(t, errLoad, true, "")
	segments, _ := mc.GetPersistentSegmentInfo(ctx, collName)
	for _, segment := range segments {
		t.Log(segment.ID)
	}

	//query
	pks := ids.(*entity.ColumnInt64).Data()
	queryResult, _ := mc.Query(
		ctx,                               // ctx
		collName,                          // CollectionName
		[]string{common.DefaultPartition}, // PartitionName
		entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10]),           // ids
		[]string{common.DefaultIntFieldName, common.DefaultFloatVecFieldName}, // OutputFields
	)
	expColumn := entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10])
	common.CheckQueryResult(t, queryResult, []entity.Column{expColumn})

}
