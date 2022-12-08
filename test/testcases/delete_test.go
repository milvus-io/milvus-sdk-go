//go:build L0

package testcases

import (
	"fmt"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"github.com/stretchr/testify/require"
)

// test delete int64 pks
func TestDelete(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/368")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// query, verify delete success
	queryRes, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Empty(t, queryRes)
}

// test delete with string pks
func TestDeleteStringPks(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/368")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createVarcharCollectionWithDataIndex(ctx, t, mc, true)

	// delete
	deleteIds := entity.NewColumnVarChar(common.DefaultVarcharFieldName, ids.(*entity.ColumnVarChar).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query, verify delete success
	queryRes, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Empty(t, queryRes)
}

// test delete from empty collection
func TestDeleteEmptyCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create
	collName := createDefaultCollection(ctx, t, mc, false)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0})
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)
}

// test delete from an not exist collection
func TestDeleteNotExistCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// flush and check row count
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1})
	errDelete := mc.DeleteByPks(ctx, "collName", common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, false, "collection collName does not exist")
}

// test delete from an not exist partition
func TestDeleteNotExistPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, "p1", deleteIds)
	common.CheckErr(t, errDelete, false, fmt.Sprintf("partition p1 of collection %s does not exist", collName))
}

// test delete empty partition names
func TestDeleteEmptyPartitionNames(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/368")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	emptyPartitionName := ""
	// create
	collName := createDefaultCollection(ctx, t, mc, false)

	// insert "" partition and flush
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, emptyPartitionName, intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	mc.Flush(ctx, collName, false)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, emptyPartitionName, deleteIds)
	common.CheckErr(t, errDelete, true)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query, verify delete success
	queryRes, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Empty(t, queryRes)
}

// test delete from empty partition
func TestDeleteEmptyPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert into default partition, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// create partition and empty
	mc.CreatePartition(ctx, collName, "p1")

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete from empty partition p1
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, "p1", deleteIds)
	common.CheckErr(t, errDelete, true)

	// query deleteIds in default partition
	queryResult, _ := mc.Query(
		ctx,
		collName,
		[]string{},
		deleteIds,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{deleteIds})
}

// test delete from partition which data not meet ids
func TestDeletePartitionIdsNotMatch(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false)
	partitionName, idsDefault, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// delete [0:10) from new partition -> delete nothing
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, idsDefault.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, partitionName, deleteIds)
	common.CheckErr(t, errDelete, true)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query deleteIds in default partition
	queryResult, _ := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		deleteIds,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{deleteIds})
}

// test delete with nil ids
func TestDeleteNilIds(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/369")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create
	collName := createDefaultCollection(ctx, t, mc, false)

	// delete
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, nil)
	common.CheckErr(t, errDelete, false, "error")
}

// test delete ids field not pk int64
func TestDeleteNotPkIds(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnFloat(common.DefaultFloatFieldName, []float32{0})
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, false, "only int64 and varchar column can be primary key for now")
}

// test delete with duplicated data ids
func TestDeleteDuplicatedPks(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/368")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 0, 0, 0, 0})
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// query, verify delete success
	queryRes, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Empty(t, queryRes)
}
