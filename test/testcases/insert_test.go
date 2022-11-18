package testcases

import (
	"context"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"github.com/stretchr/testify/require"
)

// test insert
func TestInsert(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false)

	// insert
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	common.CheckInsertResult(t, ids, intColumn)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test insert with autoID collection
func TestInsertAutoId(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, "", floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, common.DefaultNb, ids.Len())

	// list partition
	partitions, _ := mc.ShowPartitions(ctx, collName)
	require.Equal(t, partitions[0].Name, common.DefaultPartition)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test insert binary vectors
func TestInsertBinaryCollection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()

	// connect
	mc := createMilvusClient(ctx, t)

	// create binary collection with autoID true
	collName := common.GenRandomString(6)
	binaryFields := common.GenDefaultBinaryFields(true, common.DefaultDimStr)
	schema := common.GenSchema(collName, true, binaryFields)
	mc.CreateCollection(ctx, schema, common.DefaultShards)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultBinaryData(common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, "", floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, common.DefaultNb, ids.Len())

	// list partition
	partitions, _ := mc.ShowPartitions(ctx, collName)
	require.Equal(t, partitions[0].Name, common.DefaultPartition)

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test insert into an not existed partition
func TestInsertNotExistPartition(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, "aaa", floatColumn, vecColumn)
	common.CheckErr(t, errInsert, false, "does not exist")
}

// test insert data into collection that has all scala fields
func TestInsertAllFieldsData(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	mc := createMilvusClient(ctx, t)

	// prepare fields, name, schema
	allFields := common.GenAllFields()
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, allFields)

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// prepare data
	int64Values := make([]int64, 0, common.DefaultNb)
	boolValues := make([]bool, 0, common.DefaultNb)
	int8Values := make([]int8, 0, common.DefaultNb)
	int16Values := make([]int16, 0, common.DefaultNb)
	int32Values := make([]int32, 0, common.DefaultNb)
	floatValues := make([]float32, 0, common.DefaultNb)
	doubleValues := make([]float64, 0, common.DefaultNb)
	varcharValues := make([]string, 0, common.DefaultNb)
	floatVectors := make([][]float32, 0, common.DefaultNb)
	for i := 0; i < common.DefaultNb; i++ {
		int64Values = append(int64Values, int64(i))
		boolValues = append(boolValues, i/2 == 0)
		int8Values = append(int8Values, int8(i))
		int16Values = append(int16Values, int16(i))
		int32Values = append(int32Values, int32(i))
		floatValues = append(floatValues, float32(i))
		doubleValues = append(doubleValues, float64(i))
		varcharValues = append(varcharValues, strconv.Itoa(i))
		vec := make([]float32, 0, common.DefaultDim)
		for j := 0; j < common.DefaultDim; j++ {
			vec = append(vec, rand.Float32())
		}
		floatVectors = append(floatVectors, vec)
	}

	// insert data
	ids, errInsert := mc.Insert(
		ctx,
		collName,
		"",
		entity.NewColumnInt64("int64", int64Values),
		entity.NewColumnBool("bool", boolValues),
		entity.NewColumnInt8("int8", int8Values),
		entity.NewColumnInt16("int16", int16Values),
		entity.NewColumnInt32("int32", int32Values),
		entity.NewColumnFloat("float", floatValues),
		entity.NewColumnDouble("double", doubleValues),
		entity.NewColumnVarChar("varChar", varcharValues),
		entity.NewColumnFloatVector("floatVector", common.DefaultDim, floatVectors),
	)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, common.DefaultNb, ids.Len())

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test insert data columns len, order mismatch fields
func TestInsertColumnsMismatchFields(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false)
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(common.DefaultNb, common.DefaultDim)

	// len(column) < len(fields)
	_, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn)
	common.CheckErr(t, errInsert, false, "not passed")

	// len(column) > len(fields)
	_, errInsert2 := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn, floatColumn)
	common.CheckErr(t, errInsert2, false, "len(columns) mismatch the len(fields), len(columns): 4, len(fields): 3")

	// order(column) != order(fields)
	_, errInsert3 := mc.Insert(ctx, collName, "", floatColumn, vecColumn, intColumn)
	common.CheckErr(t, errInsert3, true)
}

func TestInsertColumnsDifferentLen(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false)

	// data, different column has different len
	int64Values := make([]int64, 0, 100)
	floatValues := make([]float32, 0, 200)
	vecFloatValues := make([][]float32, 0, 200)
	for i := 0; i < 100; i++ {
		int64Values = append(int64Values, int64(i))
	}
	for i := 0; i < 200; i++ {
		floatValues = append(floatValues, float32(i))
		vec := make([]float32, 0, common.DefaultDim)
		for j := 0; j < common.DefaultDim; j++ {
			vec = append(vec, rand.Float32())
		}
		vecFloatValues = append(vecFloatValues, vec)
	}

	// insert
	_, errInsert := mc.Insert(ctx, collName, "",
		entity.NewColumnInt64(common.DefaultIntFieldName, int64Values),
		entity.NewColumnFloat(common.DefaultFloatFieldName, floatValues),
		entity.NewColumnFloatVector(common.DefaultFloatVecFieldName, common.DefaultDim, vecFloatValues))
	common.CheckErr(t, errInsert, false, "column size not match")
}
