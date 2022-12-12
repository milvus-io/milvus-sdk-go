//go:build L0

package testcases

import (
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test query from default partition
func TestQueryDefaultPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pks := ids.(*entity.ColumnInt64).Data()
	queryResult, _ := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10]),
		[]string{common.DefaultIntFieldName},
	)
	expColumn := entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10])
	common.CheckQueryResult(t, queryResult, []entity.Column{expColumn})
}

// test query with varchar field filter
func TestQueryVarcharField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createVarcharCollectionWithDataIndex(ctx, t, mc, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pks := ids.(*entity.ColumnVarChar).Data()
	queryResult, _ := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnVarChar(common.DefaultVarcharFieldName, pks[:10]),
		[]string{common.DefaultVarcharFieldName},
	)
	expColumn := entity.NewColumnVarChar(common.DefaultVarcharFieldName, pks[:10])
	common.CheckQueryResult(t, queryResult, []entity.Column{expColumn})
}

// query from not existed collection
func TestQueryNotExistCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pks := ids.(*entity.ColumnInt64).Data()
	_, errQuery := mc.Query(
		ctx,
		"collName",
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10]),
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, false, "can't find collection")
}

// query from not existed partition
func TestQueryNotExistPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pks := ids.(*entity.ColumnInt64).Data()
	_, errQuery := mc.Query(
		ctx,
		collName,
		[]string{"aaa"},
		entity.NewColumnInt64(common.DefaultIntFieldName, pks[:10]),
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, false, "partition name aaa not found")
}

// test query with empty partition name
func TestQueryEmptyPartitionName(t *testing.T) {
	emptyPartitionName := ""

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create
	collName := createDefaultCollection(ctx, t, mc, false)

	// insert "" partition and flush
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, emptyPartitionName, intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	mc.Flush(ctx, collName, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query from "" partitions, expect to query from default partition
	_, errQuery := mc.Query(
		ctx,
		collName,
		[]string{emptyPartitionName},
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.Data()[:10]),
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, false, "Partition name should not be empty")
}

// query with empty partition names, actually query from all partitions
func TestQueryMultiPartitions(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/368")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query from multi partition names
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, common.DefaultNb, common.DefaultNb*2 - 1})
	queryResultMultiPartition, _ := mc.Query(ctx, collName, []string{common.DefaultPartition, partitionName}, queryIds,
		[]string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{queryIds})

	//query from empty partition names, expect to query from all partitions
	queryResultEmptyPartition, _ := mc.Query(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultEmptyPartition, []entity.Column{queryIds})

	// query from new partition and query successfully
	queryResultPartition, _ := mc.Query(ctx, collName, []string{partitionName}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition,
		[]entity.Column{entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb, common.DefaultNb*2 - 1})})

	// query from new partition and query gets empty result
	queryResultEmpty, errQuery := mc.Query(ctx, collName, []string{partitionName}, entity.NewColumnInt64(common.DefaultIntFieldName,
		[]int64{0}), []string{common.DefaultIntFieldName})
	require.Empty(t, queryResultEmpty)
	common.CheckErr(t, errQuery, true)
}

// test query with empty ids
func TestQueryEmptyIds(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/365")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{})
	queryResult, errQuery := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		queryIds,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, true)
	require.Empty(t, queryResult)
}

// test query with non-primary field filter, and output scalar fields
func TestQueryNonPrimaryFields(t *testing.T) {
	t.Skip("Issue: https://github.com/milvus-io/milvus-sdk-go/issues/366")
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create and insert
	collName, _ := createCollectionAllFields(ctx, t, mc, common.DefaultNb, 0)
	mc.Flush(ctx, collName, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, "floatVec", idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query with non-primary field
	queryIds := []entity.Column{
		entity.NewColumnBool("bool", []bool{true}),
		entity.NewColumnInt8("int8", []int8{0}),
		entity.NewColumnInt16("int16", []int16{0}),
		entity.NewColumnInt32("int32", []int32{0}),
		entity.NewColumnFloat("float", []float32{0}),
		entity.NewColumnDouble("double", []float64{0}),
	}

	for _, idsColumn := range queryIds {
		queryResultMultiPartition, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition}, idsColumn,
			[]string{common.DefaultIntFieldName})

		// TODO only int64 and varchar column can be primary key for now
		common.CheckErr(t, errQuery, true)
		common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0})})
	}
}

// test query empty or one scalar output fields
func TestQueryEmptyOutputFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query with empty output fields -> output "int64"
	queryEmptyOutputs, _ := mc.Query(
		ctx, collName, []string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
		[]string{},
	)
	common.CheckOutputFields(t, queryEmptyOutputs, []string{common.DefaultIntFieldName})

	// query with "float" output fields -> output "int64, float"
	queryFloatOutputs, _ := mc.Query(
		ctx, collName, []string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
		[]string{common.DefaultFloatFieldName},
	)
	common.CheckOutputFields(t, queryFloatOutputs, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName})
}

// test query output int64 and float and floatVector fields
func TestQueryOutputFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert data into default partition, pks from 0 to DefaultNb
	collName := createDefaultCollection(ctx, t, mc, false)
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	mc.Flush(ctx, collName, false)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pos := 10
	queryResult, _ := mc.Query(
		ctx, collName,
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.Data()[:pos]),
		[]string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.Data()[:pos]),
		entity.NewColumnFloat(common.DefaultFloatFieldName, floatColumn.Data()[:pos]),
		entity.NewColumnFloatVector(common.DefaultFloatVecFieldName, common.DefaultDim, vecColumn.Data()[:pos]),
	})
	common.CheckOutputFields(t, queryResult, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName})
}

// test query output varchar and binaryVector fields
func TestQueryOutputBinaryAndVarchar(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert data into default partition, pks from 0 to DefaultNb
	collName := createDefaultVarcharCollection(ctx, t, mc)
	varcharColumn, vecColumn := common.GenDefaultVarcharData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, "", varcharColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	mc.Flush(ctx, collName, false)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])

	// create index
	idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
	err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName(""))
	common.CheckErr(t, err, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	pos := 10
	queryResult, _ := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn.Data()[:pos]),
		[]string{common.DefaultBinaryVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn.Data()[:pos]),
		entity.NewColumnBinaryVector(common.DefaultBinaryVecFieldName, common.DefaultDim, vecColumn.Data()[:pos]),
	})
	common.CheckOutputFields(t, queryResult, []string{common.DefaultBinaryVecFieldName, common.DefaultVarcharFieldName})
}

// test query output all scalar fields
func TestOutputAllScalarFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create and insert
	collName, _ := createCollectionAllFields(ctx, t, mc, common.DefaultNb, 0)
	mc.Flush(ctx, collName, false)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, "floatVec", idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query with non-primary field
	queryIds := []entity.Column{
		entity.NewColumnBool("bool", []bool{true}),
		entity.NewColumnInt8("int8", []int8{0}),
		entity.NewColumnInt16("int16", []int16{0}),
		entity.NewColumnInt32("int32", []int32{0}),
		entity.NewColumnFloat("float", []float32{0}),
		entity.NewColumnDouble("double", []float64{0}),
	}

	queryResultAllScalar, errQuery := mc.Query(ctx, collName, []string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0}), []string{"bool", "int8", "int16", "int32", "float", "double"})
	common.CheckErr(t, errQuery, true)
	common.CheckQueryResult(t, queryResultAllScalar, queryIds)
	common.CheckOutputFields(t, queryResultAllScalar, []string{"int64", "bool", "int8", "int16", "int32", "float", "double"})
}

// test query with an not existed field
func TestQueryNotExistField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	_, errQuery := mc.Query(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
		[]string{common.DefaultIntFieldName, "varchar"},
	)
	common.CheckErr(t, errQuery, false, "field varchar not exist")
}

// TODO offset and limit
// TODO consistency level
