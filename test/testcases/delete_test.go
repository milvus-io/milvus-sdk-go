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
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true, client.WithConsistencyLevel(entity.ClStrong))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// query, verify delete success
	queryRes, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Zero(t, queryRes[0].Len())
}

// test delete with string pks
func TestDeleteStringPks(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createVarcharCollectionWithDataIndex(ctx, t, mc, true, client.WithConsistencyLevel(entity.ClStrong))

	// delete
	deleteIds := entity.NewColumnVarChar(common.DefaultVarcharFieldName, ids.(*entity.ColumnVarChar).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query, verify delete success
	queryRes, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Zero(t, queryRes[0].Len())
}

// test delete from empty collection
func TestDeleteEmptyCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

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
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	emptyPartitionName := ""
	// create
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithConsistencyLevel(entity.ClStrong))

	// insert "" partition and flush
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, emptyPartitionName, intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	mc.Flush(ctx, collName, false)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, emptyPartitionName, deleteIds)
	common.CheckErr(t, errDelete, true)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query, verify delete success
	queryRes, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Zero(t, queryRes[0].Len())
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
	queryResult, _ := mc.QueryByPks(
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
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, vecColumnDefault, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// delete [0:10) from new partition -> delete nothing
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, vecColumnDefault.IdsColumn.(*entity.ColumnInt64).Data()[:10])
	errDelete := mc.DeleteByPks(ctx, collName, partitionName, deleteIds)
	common.CheckErr(t, errDelete, true)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query deleteIds in default partition
	queryResult, _ := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		deleteIds,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{deleteIds})
}

// test delete with nil ids
func TestDeleteEmptyIds(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

	// delete
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, entity.NewColumnInt64(common.DefaultIntFieldName, []int64{}))
	common.CheckErr(t, errDelete, false, "ids len must not be zero")
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
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true, client.WithConsistencyLevel(entity.ClStrong))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// delete
	deleteIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 0, 0, 0, 0})
	errDelete := mc.DeleteByPks(ctx, collName, common.DefaultPartition, deleteIds)
	common.CheckErr(t, errDelete, true)

	// query, verify delete success
	queryRes, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition}, deleteIds, []string{})
	common.CheckErr(t, errQuery, true)
	require.Zero(t, queryRes[0].Len())
}

func TestDeleteExpressions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: AllFields,
		AutoID:               false,
		EnableDynamicField:   true,
		ShardsNum:            common.DefaultShards,
		Dim:                  common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

	// insert
	dp := DataParams{
		CollectionName:       collName,
		PartitionName:        "",
		CollectionFieldsType: AllFields,
		start:                0,
		nb:                   common.DefaultNb,
		dim:                  common.DefaultDim,
		EnableDynamicField:   true,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	for _, field := range common.AllVectorsFieldsName {
		err := mc.CreateIndex(ctx, collName, field, idx, false)
		common.CheckErr(t, err, true)
	}

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query with different expr and count
	exprCounts := []string{
		// pk int64 field expr: < in && ||
		fmt.Sprintf("%s < 1000", common.DefaultIntFieldName),
		fmt.Sprintf("%s in [0, 1, 2]", common.DefaultIntFieldName),
		fmt.Sprintf("%s >= 1000 && %s < 2000", common.DefaultIntFieldName, common.DefaultIntFieldName),
		fmt.Sprintf("%s >= 1000 || %s > 2000", common.DefaultIntFieldName, common.DefaultIntFieldName),
		fmt.Sprintf("%s < 1000", common.DefaultFloatFieldName),

		// json and dynamic field filter expr: == < in bool/ list/ int
		fmt.Sprintf("%s['number'] == 0", common.DefaultJSONFieldName),
		fmt.Sprintf("%s['number'] < 100 and %s['number'] != 0", common.DefaultJSONFieldName, common.DefaultJSONFieldName),
		fmt.Sprintf("%s < 100", common.DefaultDynamicNumberField),
		"dynamicNumber % 2 == 0",
		fmt.Sprintf("%s['bool'] == true", common.DefaultJSONFieldName),
		fmt.Sprintf("%s == false", common.DefaultDynamicBoolField),
		fmt.Sprintf("%s in ['1', '2'] ", common.DefaultDynamicStringField),
		fmt.Sprintf("%s['string'] in ['1', '2', '5'] ", common.DefaultJSONFieldName),
		fmt.Sprintf("%s['list'] == [1, 2] ", common.DefaultJSONFieldName),
		fmt.Sprintf("%s['list'] == [0, 1] ", common.DefaultJSONFieldName),
		fmt.Sprintf("%s['list'][0] < 10 ", common.DefaultJSONFieldName),
		fmt.Sprintf("%s[\"dynamicList\"] != [2, 3]", common.DefaultDynamicFieldName),
		fmt.Sprintf("%s > 2500", common.DefaultJSONFieldName),
		fmt.Sprintf("%s > 2000.5", common.DefaultJSONFieldName),
		fmt.Sprintf("%s[0] == 2503", common.DefaultJSONFieldName),
		fmt.Sprintf("%s like '21%%' ", common.DefaultJSONFieldName),

		// json contains
		fmt.Sprintf("json_contains (%s['list'], 2)", common.DefaultJSONFieldName),
		fmt.Sprintf("json_contains (%s['number'], 0)", common.DefaultJSONFieldName),
		fmt.Sprintf("json_contains_all (%s['list'], [1, 2])", common.DefaultJSONFieldName),
		fmt.Sprintf("JSON_CONTAINS_ANY (%s['list'], [1, 3])", common.DefaultJSONFieldName),
		// string like
		"dynamicString like '1%' ",

		// key exist
		fmt.Sprintf("exists %s['list']", common.DefaultJSONFieldName),
		fmt.Sprintf("exists a "),
		fmt.Sprintf("exists %s ", common.DefaultDynamicListField),
		fmt.Sprintf("exists %s ", common.DefaultDynamicStringField),
		// data type not match and no error
		fmt.Sprintf("%s['number'] == '0' ", common.DefaultJSONFieldName),
	}

	for _, _exprCount := range exprCounts {
		err := mc.Delete(ctx, collName, "", _exprCount)
		common.CheckErr(t, err, true)

		// query
		countRes, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, _exprCount, []string{common.QueryCountFieldName})
		require.Equal(t, int64(0), countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
	}
}

func TestDeleteInvalidExpr(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecJSON,
		AutoID:               false,
		EnableDynamicField:   true,
		ShardsNum:            common.DefaultShards,
		Dim:                  common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName:       collName,
		PartitionName:        "",
		CollectionFieldsType: Int64FloatVecJSON,
		start:                0,
		nb:                   common.DefaultNb,
		dim:                  common.DefaultDim,
		EnableDynamicField:   true,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	err := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	err = mc.Delete(ctx, collName, "", "")
	common.CheckErr(t, err, false, "invalid expression: invalid parameter")

	for _, _invalidExprs := range common.InvalidExpressions {
		err := mc.Delete(ctx, collName, "", _invalidExprs.Expr)
		common.CheckErr(t, err, _invalidExprs.ErrNil, _invalidExprs.ErrMsg)
	}
}

func TestDeleteComplexExprWithoutLoading(t *testing.T) {
	// TODO
	ctx := createContext(t, time.Second*common.DefaultTimeout*3)
	// connect
	mc := createMilvusClient(ctx, t)

	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

	// prepare and insert data
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp)
	mc.Flush(ctx, collName, false)

	err := mc.Delete(ctx, collName, "", "int64 < 100")
	common.CheckErr(t, err, false, "collection not loaded")

	// load
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 72)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	err = mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, err, true)

	err = mc.Delete(ctx, collName, "", "int64 < 100")
	common.CheckErr(t, err, true)
}
