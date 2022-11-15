package testcases

import (
	"context"
	"flag"
	"log"
	"os"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

var addr = flag.String("addr", "localhost:19530", "server host and port")

func init() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
}

// teardown
func teardown() {
	log.Println("Start to tear down all")
	ctx := context.Background()
	mc, err := base.NewMilvusClient(ctx, *addr)
	if err != nil {
		log.Fatalf("teardown failed to connect milvus with error %v", err)
	}
	defer mc.Close()
	collections, _ := mc.ListCollections(ctx)
	for _, collection := range collections {
		mc.DropCollection(ctx, collection.Name)
	}
}

// create connect
func createMilvusClient(ctx context.Context, t *testing.T) *base.MilvusClient {
	t.Helper()

	mc, err := base.NewMilvusClient(ctx, *addr)
	common.CheckErr(t, err, true, "")

	t.Cleanup(func() {
		mc.Close()
	})

	return mc
}

// create default collection
func createDefaultCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultFields()
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true, "")

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
		//mc.Close()
	})
	return collName
}

func createCollectionWithDataAndIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultCollection(ctx, t, mc, autoID)

	// insert data
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb)
	ids, errInsert := mc.Insert(
		context.Background(),    // ctx
		collName,                // CollectionName
		common.DefaultPartition, // partitionName
		intColumn,               // columnarData
		floatColumn,             // columnarData
		vecColumn,               // columnarData
	)
	common.CheckErr(t, errInsert, true, "")
	common.CheckInsertResult(t, ids, intColumn)

	// flush
	mc.Flush(ctx, collName, false)

	// TODO create index
	if withIndex {
		//mc.CreateIndex()
	}
	return collName, ids

}

func TestMain(m *testing.M) {
	flag.Parse()
	log.Printf("parse addr=%s", *addr)
	code := m.Run()
	teardown()
	os.Exit(code)
}
