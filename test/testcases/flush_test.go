//go:build L0

package testcases

import (
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"github.com/stretchr/testify/require"
)

// test flush collection
func TestFlushCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

	// insert
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	common.CheckInsertResult(t, ids, intColumn)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test flush empty collection
func TestFlushEmptyCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, "0", stats[common.RowCount])
}

// test flush not existed collection
func TestFlushNotExistedCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// flush and check row count
	errFlush := mc.Flush(ctx, "collName", false)
	common.CheckErr(t, errFlush, false, "collection not found")
}

// test flush async
func TestFlushAsync(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

	// insert
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	common.CheckInsertResult(t, ids, intColumn)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, true)
	common.CheckErr(t, errFlush, true)
	mc.GetCollectionStatistics(ctx, collName)

	// wait flush done
	for {
		time.Sleep(time.Second * 1)
		stats, errStatist := mc.GetCollectionStatistics(ctx, collName)
		if errStatist == nil {
			if strconv.Itoa(common.DefaultNb) == stats[common.RowCount] {
				break
			}
		} else {
			t.FailNow()
		}
	}
}
