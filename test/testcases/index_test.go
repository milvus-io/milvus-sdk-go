package testcases

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"testing"
	"time"
)

func TestCreateIndex(t *testing.T) {
	ctx, _ := context.WithTimeout(context.Background(), time.Second * common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false)

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true, "")

	//create index
	//idx, _ := entity.NewIndexHNSW(   // NewIndex func
	//	entity.L2,                        // metricType
	//	20,                            // ConstructParams
	//	300,                  //
	//)
	//errIndex := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, true)
	//assert.Nil(t, errIndex)
	//
	//indexes, _ := mc.DescribeIndex(ctx, collName, common.DefaultFloatVecFieldName)
	//log.Print(indexes)
	//for _, index := range indexes {
	//	log.Println(index.Name())
	//	log.Println(index.IndexType())
	//	log.Println(index.Params())
	//}
}
