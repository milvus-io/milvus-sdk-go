package testcases

import (
	"context"
	"flag"
	"log"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

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
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	mc, err := base.NewDefaultMilvusClient(ctx, *addr)
	if err != nil {
		log.Fatalf("teardown failed to connect milvus with error %v", err)
	}
	defer mc.Close()
	collections, _ := mc.ListCollections(ctx)
	for _, collection := range collections {
		mc.DropCollection(ctx, collection.Name)
	}
}

//
func createContext(t *testing.T, timeout time.Duration) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	t.Cleanup(func() {
		cancel()
	})
	return ctx
}

// create connect
func createMilvusClient(ctx context.Context, t *testing.T) *base.MilvusClient {
	t.Helper()

	mc, err := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, err, true)

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
	fields := common.GenDefaultFields(autoID)
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
	})
	return collName
}

// create default collection
func createDefaultBinaryCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, dim string) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultBinaryFields(autoID, dim)
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
		//mc.Close()
	})
	return collName
}

// create default varchar pk collection
func createDefaultVarcharCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultVarcharFields(false)
	schema := common.GenSchema(collName, false, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
		//mc.Close()
	})
	return collName
}

func createCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultCollection(ctx, t, mc, autoID)

	// insert data
	var ids entity.Column
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultDim)
	if autoID {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		ids = pk
	} else {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, intColumn, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		common.CheckInsertResult(t, pk, intColumn)
		ids = pk
	}

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, false, idx, client.WithIndexName(""))
		common.CheckErr(t, err, true)
	}
	return collName, ids
}

func createBinaryCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultBinaryCollection(ctx, t, mc, autoID, common.DefaultDimStr)

	// insert data
	var ids entity.Column
	intColumn, floatColumn, vecColumn := common.GenDefaultBinaryData(common.DefaultNb, common.DefaultDim)
	if autoID {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		ids = pk
	} else {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, intColumn, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		common.CheckInsertResult(t, pk, intColumn)
		ids = pk
	}

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, false, idx, client.WithIndexName(""))
		common.CheckErr(t, err, true)
	}
	return collName, ids
}

func createVarcharCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultVarcharCollection(ctx, t, mc)

	// insert data
	varcharColumn, vecColumn := common.GenDefaultVarcharData(common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, varcharColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	common.CheckInsertResult(t, ids, varcharColumn)

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, false, idx, client.WithIndexName(""))
		common.CheckErr(t, err, true)
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
