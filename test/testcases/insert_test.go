package testcases

import (
	"context"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"github.com/stretchr/testify/require"
)

func TestInsert(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false)

	// insert
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb)
	ids, errInsert := mc.Insert(
		context.Background(), // ctx
		collName,             // CollectionName
		"",                   // partitionName
		intColumn,            // columnarData
		floatColumn,          // columnarData
		vecColumn,            // columnarData
	)
	common.CheckErr(t, errInsert, true, "")
	common.CheckInsertResult(t, ids, intColumn)
	log.Println(ids.Len())
}

// TODO issue: field int64 not passed
func TestInsertAutoId(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, true)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb)
	ids, errInsert := mc.Insert(
		context.Background(), // ctx
		collName,             // CollectionName
		"",                   // partitionName
		floatColumn,          // columnarData
		vecColumn,            // columnarData
	)
	common.CheckErr(t, errInsert, true, "")
	require.Equal(t, common.DefaultNb, ids.Len())
}
