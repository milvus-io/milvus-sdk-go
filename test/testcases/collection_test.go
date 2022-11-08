package testcases

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"log"
	"testing"
	"time"
)

func TestCreateCollection(t *testing.T) {
	ctx, _ := context.WithTimeout(context.Background(), time.Second * common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// prepare
	collName := common.GenRandomString(6)
	fields := common.GenDefaultFields()
	schema := common.GenSchema(collName, true, fields)
	log.Print(schema.AutoID)

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true, "")

	// check describe collection
	collection, _ := mc.DescribeCollection(ctx, collName)
	common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)
	log.Println(collection.Schema.Fields[2].IndexParams)
	log.Println(schema.Fields[2].IndexParams)

	// check collName in ListCollections
	collections, errListCollection := mc.ListCollections(ctx)
	common.CheckErr(t, errListCollection, true, "")
	common.CheckContainsCollection(t, collections, collName)
}