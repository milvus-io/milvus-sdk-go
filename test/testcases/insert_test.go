//go:build L0

package testcases

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"github.com/stretchr/testify/require"
)

// test insert
func TestInsert(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamicField := range []bool{true, false} {
		// create default collection
		collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithEnableDynamicSchema(enableDynamicField))

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
}

// test insert with autoID collection
func TestInsertAutoId(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
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

func TestInsertAutoIdPkData(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)

	// insert
	pkColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, "", pkColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, false, "invalid parameter")

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, "0", stats[common.RowCount])
}

// test insert binary vectors
func TestInsertBinaryCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create binary collection with autoID true
	collName := common.GenRandomString(6)
	binaryFields := common.GenDefaultBinaryFields(true, common.DefaultDim)
	schema := common.GenSchema(collName, true, binaryFields)
	mc.CreateCollection(ctx, schema, common.DefaultShards)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultBinaryData(0, common.DefaultNb, common.DefaultDim)
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

// test insert not exist collection
func TestInsertNotExistCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// insert
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, "notExist", "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, false, "can't find collection")
}

// test insert into an not existed partition
func TestInsertNotExistPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)

	// insert
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errInsert := mc.Insert(ctx, collName, "aaa", floatColumn, vecColumn)
	common.CheckErr(t, errInsert, false, "partition not found")
}

// test insert data into collection that has all scala fields
func TestInsertAllFieldsData(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// prepare fields, name, schema
	allFields := common.GenAllFields()
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, allFields, common.WithEnableDynamicField(true))

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// prepare and insert data
	data := common.GenAllFieldsData(0, common.DefaultNb, common.DefaultDim)
	data = append(data, common.GenDynamicFieldData(0, common.DefaultNb)...)
	// insert data
	ids, errInsert := mc.Insert(
		ctx,
		collName,
		"",
		data...,
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
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)

	// len(column) < len(fields)
	_, errInsert := mc.Insert(ctx, collName, "", intColumn, floatColumn)
	common.CheckErr(t, errInsert, false, "not passed")

	// len(column) > len(fields)
	_, errInsert2 := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn, floatColumn)
	common.CheckErr(t, errInsert2, false, "duplicated column")

	//
	binaryColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeBinaryVector, "binaryVec", common.WithVectorDim(common.DefaultDim))
	_, errInsert4 := mc.Insert(ctx, collName, "", intColumn, floatColumn, vecColumn, binaryColumn)
	common.CheckErr(t, errInsert4, false, "does not exist")

	// order(column) != order(fields)
	_, errInsert3 := mc.Insert(ctx, collName, "", floatColumn, vecColumn, intColumn)
	common.CheckErr(t, errInsert3, true)
}

// test insert with columns which has different len
func TestInsertColumnsDifferentLen(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

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
		for j := 0; j < int(common.DefaultDim); j++ {
			vec = append(vec, rand.Float32())
		}
		vecFloatValues = append(vecFloatValues, vec)
	}

	// insert
	_, errInsert := mc.Insert(ctx, collName, "",
		entity.NewColumnInt64(common.DefaultIntFieldName, int64Values),
		entity.NewColumnFloat(common.DefaultFloatFieldName, floatValues),
		entity.NewColumnFloatVector(common.DefaultFloatVecFieldName, int(common.DefaultDim), vecFloatValues))
	common.CheckErr(t, errInsert, false, "column size not match")
}

// test insert rows enable or disable dynamic field
func TestInsertRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)
	for _, enableDynamicField := range []bool{true, false} {
		// create collection enable dynamic field
		schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false), common.WithEnableDynamicField(enableDynamicField))
		createCustomerCollection(ctx, t, mc, schema, common.DefaultShards)

		// insert rows
		rows := common.GenDefaultRows(0, common.DefaultNb, common.DefaultDim, enableDynamicField)
		ids, err := mc.InsertRows(ctx, schema.CollectionName, "", rows)
		common.CheckErr(t, err, true)

		int64Values := make([]int64, 0, common.DefaultNb)
		for i := 0; i < common.DefaultNb; i++ {
			int64Values = append(int64Values, int64(i))
		}
		common.CheckInsertResult(t, ids, entity.NewColumnInt64(common.DefaultIntFieldName, int64Values))

		// flush and check row count
		errFlush := mc.Flush(ctx, schema.CollectionName, false)
		common.CheckErr(t, errFlush, true)
		stats, _ := mc.GetCollectionStatistics(ctx, schema.CollectionName)
		require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
	}
}

func TestInsertDisableAutoIDRow(t *testing.T) {
	/*
		autoID: false
		- pass pk value -> insert success
		- no pk value -> error
	*/
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false))
	createCustomerCollection(ctx, t, mc, schema, common.DefaultShards)

	// pass pk value
	start := 0
	rowsWithPk := common.GenDefaultRows(start, 10, common.DefaultDim, false)
	idsWithPk, err := mc.InsertRows(ctx, schema.CollectionName, "", rowsWithPk)
	common.CheckErr(t, err, true)
	require.Contains(t, idsWithPk.(*entity.ColumnInt64).Data(), int64(start))

	// no pk value -> now error
	type tmpRow struct {
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec,omitempty" milvus:"name:floatVec"`
	}
	rowsWithoutPk := make([]interface{}, 0, 10)

	// BaseRow generate insert rows
	for i := 0; i < 10; i++ {
		baseRow := tmpRow{
			Float:    float32(i),
			FloatVec: common.GenFloatVector(common.DefaultDim),
		}
		rowsWithoutPk = append(rowsWithoutPk, &baseRow)
	}
	_, err1 := mc.InsertRows(ctx, schema.CollectionName, "", rowsWithoutPk)
	common.CheckErr(t, err1, false, "row 0 does not has field int64")
}

func TestInsertEnableAutoIDRow(t *testing.T) {
	/*
		autoID: true
		- pass pk value -> ignore passed value and auto-gen pk
		- no pk value -> insert success
	*/
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	schema := common.GenSchema(common.GenRandomString(6), true, common.GenDefaultFields(true))
	createCustomerCollection(ctx, t, mc, schema, common.DefaultShards)

	coll, _ := mc.DescribeCollection(ctx, schema.CollectionName)
	log.Println(coll.Schema.AutoID)

	// pass pk value -> ignore passed pks
	start := 0
	rowsWithPk := common.GenDefaultRows(start, 10, common.DefaultDim, false)
	idsWithPk, err := mc.InsertRows(ctx, schema.CollectionName, "", rowsWithPk)
	common.CheckErr(t, err, true)
	log.Printf("origin first rowsWithPk: %v", rowsWithPk[0])
	log.Println(idsWithPk.FieldData())
	require.NotContains(t, idsWithPk.(*entity.ColumnInt64).Data(), int64(start))

	// no pk value
	rowsWithoutPk := make([]interface{}, 0, 10)
	type tmpRow struct {
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
	}

	// BaseRow generate insert rows
	for i := 0; i < 10; i++ {
		baseRow := tmpRow{
			Float:    float32(i),
			FloatVec: common.GenFloatVector(common.DefaultDim),
		}
		rowsWithoutPk = append(rowsWithoutPk, &baseRow)
	}

	idsWithoutPk, err1 := mc.InsertRows(ctx, schema.CollectionName, "", rowsWithoutPk)
	common.CheckErr(t, err1, true)
	require.Equal(t, 10, int(idsWithoutPk.Len()))
}

// test insert json rows field name not match
func TestInsertJsonCollectionFieldNotMatch(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)
	nb := 1000

	// create json collection with random json field name
	collName := common.GenRandomString(6)
	jsonRandomField := common.GenField("", entity.FieldTypeJSON)
	fields := common.GenDefaultFields(true)
	fields = append(fields, jsonRandomField)
	schema := common.GenSchema(collName, true, fields)

	// create collection
	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, true)

	// insert data without "json" field
	rows := common.GenDefaultRows(0, nb, common.DefaultDim, true)
	_, errInsert := mc.InsertRows(ctx, collName, "", rows)
	common.CheckErr(t, errInsert, false, "does not has field")
}

// test insert json collection
func TestInsertJsonCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)
	nb := 1000

	// create collection with json field named "json"
	jsonField := common.GenField(common.DefaultJSONFieldName, entity.FieldTypeJSON)
	fields1 := common.GenDefaultFields(true)
	fields1 = append(fields1, jsonField)
	collName := common.GenRandomString(4)
	err := mc.CreateCollection(ctx, common.GenSchema(collName, true, fields1), common.DefaultShards)
	common.CheckErr(t, err, true)

	// insert rows to json collection
	rows := common.GenDefaultJSONRows(0, nb, common.DefaultDim, true)
	_, ok := rows[0].([]byte)
	if !ok {
		log.Printf("invalid type, expected []byte, got %T", rows)
	}
	_, errInsert := mc.InsertRows(ctx, collName, "", rows)
	common.CheckErr(t, errInsert, true)

	// insert json data column
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, nb, common.DefaultDim)
	jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, nb)
	ids, errInsert := mc.Insert(ctx, collName, "", floatColumn, vecColumn, jsonColumn)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(nb*2), stats[common.RowCount])

	// insert json data column less than other column
	_, floatColumn, vecColumn = common.GenDefaultColumnData(0, nb, common.DefaultDim)
	jsonColumn = common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, nb/2)
	ids, errInsert = mc.Insert(ctx, collName, "", floatColumn, vecColumn, jsonColumn)
	common.CheckErr(t, errInsert, false, "column size not match")
}

// Test insert with dynamic field
func TestInsertDynamicFieldData(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)
	nb := 1000

	// create collection enable dynamic field
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false), common.WithEnableDynamicField(true))
	createCustomerCollection(ctx, t, mc, schema, common.DefaultShards)

	// insert without dynamic field data
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, nb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, schema.CollectionName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())

	// insert with extra varchar column data
	intColumn, floatColumn, vecColumn = common.GenDefaultColumnData(nb, nb, common.DefaultDim)
	varcharColumn, _ := common.GenDefaultVarcharData(nb, nb, 0)
	ids, errInsert = mc.Insert(ctx, schema.CollectionName, "", intColumn, floatColumn, vecColumn, varcharColumn)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())

	// insert with extra int64 column data
	intColumn, floatColumn, vecColumn = common.GenDefaultColumnData(nb, nb, common.DefaultDim)
	int64Values := make([]int64, 0, nb)
	for i := 0; i < nb; i++ {
		int64Values = append(int64Values, int64(i*10))
	}
	ids, errInsert = mc.Insert(ctx, schema.CollectionName, "", intColumn, floatColumn, vecColumn, entity.NewColumnInt64("aa", int64Values))
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())

	// flush and check row count
	errFlush := mc.Flush(ctx, schema.CollectionName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, schema.CollectionName)
	require.Equal(t, strconv.Itoa(nb*3), stats[common.RowCount])

	// insert dynamic by rows struct
	rows := common.GenDefaultRows(nb*2, nb, common.DefaultDim, false)
	_, err := mc.InsertRows(context.Background(), schema.CollectionName, "", rows)
	common.CheckErr(t, err, true)

	// flush and check row count
	errFlush = mc.Flush(ctx, schema.CollectionName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ = mc.GetCollectionStatistics(ctx, schema.CollectionName)
	require.Equalf(t, strconv.Itoa(nb*4), stats[common.RowCount], fmt.Sprintf("Expected row_count: %d, actual: %s", common.DefaultNb, stats[common.RowCount]))

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	err = mc.CreateIndex(ctx, schema.CollectionName, common.DefaultFloatVecFieldName, idx, false)
	common.CheckErr(t, err, true)

	// insert column data less other column
	intColumn, floatColumn, vecColumn = common.GenDefaultColumnData(nb, nb, common.DefaultDim)
	varcharColumn, _ = common.GenDefaultVarcharData(nb, nb/2, 0)
	ids, errInsert = mc.Insert(ctx, schema.CollectionName, "", intColumn, floatColumn, vecColumn, varcharColumn)
	common.CheckErr(t, errInsert, false, "column size not match")

}

// dynamic field name is same as other field name
func TestInsertRepeatedDynamicField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)
	nb := 1000

	// create collection enable dynamic field
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false), common.WithEnableDynamicField(true))
	createCustomerCollection(ctx, t, mc, schema, 1)

	// insert column with repeated dynamic field name
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, nb, common.DefaultDim)
	dynamicColumn := common.GenColumnData(nb, nb, entity.FieldTypeInt64, common.DefaultFloatFieldName)
	_, errInsert := mc.Insert(ctx, schema.CollectionName, "", intColumn, floatColumn, vecColumn, dynamicColumn)
	common.CheckErr(t, errInsert, false, "duplicated column float found")

	// insert rows with repeated dynamic field name
	type DynamicRows struct {
		Float float32 `json:"float" milvus:"name:float"`
	}

	type dataRows struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
		DynamicRows
	}
	rows := make([]interface{}, 0, 100)
	for i := 0; i < 100; i++ {
		row := dataRows{
			Int64:    int64(i),
			Float:    float32(i),
			FloatVec: common.GenFloatVector(common.DefaultDim),
			DynamicRows: DynamicRows{
				Float: 0.0,
			},
		}
		rows = append(rows, row)
	}

	_, err := mc.InsertRows(context.Background(), schema.CollectionName, "", rows)
	common.CheckErr(t, err, false, "column has duplicated name: float when parsing field: DynamicRows")
}

// test insert array column with empty data
func TestInsertEmptyArray(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecArray, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	var capacity int64
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecArray,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
}

// test insert array type rows data
func TestInsertArrayRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	for _, dynamic := range []bool{true, false} {
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: dynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity}
		collName := createCollection(ctx, t, mc, cp)

		// prepare and insert array rows data
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: dynamic, WithRows: true}
		_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))

		// flush and check row count
		errFlush := mc.Flush(ctx, collName, false)
		common.CheckErr(t, errFlush, true)
		stats, _ := mc.GetCollectionStatistics(ctx, collName)
		require.Equal(t, strconv.Itoa(common.DefaultNb), stats[common.RowCount])
	}
}

// test insert array data type not match array field element type
func TestInsertArrayDataTypeNotMatch(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, arrayType := range common.ArrayFieldType {

		// fields: int64 + float + vector + array with TestCapacity
		defaultFields := common.GenDefaultFields(false)
		arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray,
			common.WithElementType(arrayType), common.WithMaxCapacity(100), common.WithMaxLength(100))
		fields := append(defaultFields, arrayField)

		// create collection
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, false, fields, common.WithEnableDynamicField(true))
		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, true)

		// insert data type does not match schema array element type
		var columnType entity.FieldType
		if arrayType == entity.FieldTypeInt64 {
			columnType = entity.FieldTypeBool
		} else {
			columnType = entity.FieldTypeInt64
		}
		intColumn, floatColumn, vectorColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
		arrayColumn := common.GenArrayColumnData(0, common.DefaultNb, common.DefaultArrayFieldName,
			common.WithArrayElementType(columnType))
		_, err = mc.Insert(ctx, collName, "", intColumn, floatColumn, vectorColumn, arrayColumn)
		common.CheckErr(t, err, false, "insert data does not match")
	}
}

// test insert array column data that capacity exceeds max capacity
func TestInsertArrayDataCapacityExceed(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, arrayType := range common.ArrayFieldType {
		// fields: int64 + float + vector + array with TestCapacity
		defaultFields := common.GenDefaultFields(false)
		arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray,
			common.WithElementType(arrayType), common.WithMaxCapacity(common.TestCapacity), common.WithMaxLength(100))
		fields := append(defaultFields, arrayField)

		// create collection
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, false, fields, common.WithEnableDynamicField(true))
		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, true)

		// insert data capacity larger than TestCapacity
		intColumn, floatColumn, vectorColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
		arrayColumn := common.GenArrayColumnData(0, common.DefaultNb, common.DefaultArrayFieldName,
			common.WithArrayElementType(arrayType), common.WithArrayCapacity(common.TestCapacity+1))
		_, err = mc.Insert(ctx, collName, "", intColumn, floatColumn, vectorColumn, arrayColumn)
		common.CheckErr(t, err, false, "array length exceeds max capacity")
	}
}

// test insert sparse vector column and rows
func TestInsertSparseData(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	cp := CollectionParams{CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen}
	collName := createCollection(ctx, t, mc, cp)

	// insert data column
	intColumn1 := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeInt64, common.DefaultIntFieldName)
	data := []entity.Column{
		intColumn1,
		common.GenColumnData(0, common.DefaultNb, entity.FieldTypeVarChar, common.DefaultVarcharFieldName),
		common.GenColumnData(0, common.DefaultNb, entity.FieldTypeFloatVector, common.DefaultFloatVecFieldName, common.WithVectorDim(common.DefaultDim)),
		common.GenColumnData(0, common.DefaultNb, entity.FieldTypeSparseVector, common.DefaultSparseVecFieldName, common.WithSparseVectorLen(20)),
	}
	ids, err := mc.Insert(ctx, collName, "", data...)
	common.CheckErr(t, err, true)
	common.CheckInsertResult(t, ids, intColumn1)

	// insert rows
	rows := common.GenDefaultSparseRows(common.DefaultNb, common.DefaultNb, common.DefaultDim, 50, true)
	ids2, err := mc.InsertRows(ctx, collName, "", rows)
	common.CheckErr(t, err, true)
	require.Equal(t, ids2.Len(), common.DefaultNb)

	// flush and verify
	err = mc.Flush(ctx, collName, false)
	common.CheckErr(t, err, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(common.DefaultNb*2), stats[common.RowCount])
}

// the dimension of a sparse embedding can be any value from 0 to (maximum of uint32 - 1)
func TestInsertSparseDataMaxDim(t *testing.T) {
	// invalid sparse vector: positions >= uint32
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	cp := CollectionParams{CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen}
	collName := createCollection(ctx, t, mc, cp)

	// insert data column
	pkColumn := common.GenColumnData(0, 1, entity.FieldTypeInt64, common.DefaultIntFieldName)
	data := []entity.Column{
		pkColumn,
		common.GenColumnData(0, 1, entity.FieldTypeVarChar, common.DefaultVarcharFieldName),
		common.GenColumnData(0, 1, entity.FieldTypeFloatVector, common.DefaultFloatVecFieldName, common.WithVectorDim(common.DefaultDim)),
	}
	// sparse vector with max dim
	positions := []uint32{0, math.MaxUint32 - 10, math.MaxUint32 - 1}
	values := []float32{0.453, 5.0776, 100.098}
	sparseVec, err := entity.NewSliceSparseEmbedding(positions, values)
	common.CheckErr(t, err, true)
	data = append(data, entity.NewColumnSparseVectors(common.DefaultSparseVecFieldName, []entity.SparseEmbedding{sparseVec}))
	ids, err := mc.Insert(ctx, collName, "", data...)
	common.CheckErr(t, err, true)
	common.CheckInsertResult(t, ids, pkColumn)
}

func TestInsertSparseInvalidVector(t *testing.T) {
	// invalid sparse vector: len(positions) != len(values)
	positions := []uint32{1, 10}
	values := []float32{0.4, 5.0, 0.34}
	_, err := entity.NewSliceSparseEmbedding(positions, values)
	common.CheckErr(t, err, false, "invalid sparse embedding input, positions shall have same number of values")

	// invalid sparse vector: positions >= uint32
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	cp := CollectionParams{CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen}
	collName := createCollection(ctx, t, mc, cp)

	// insert data column
	data := []entity.Column{
		common.GenColumnData(0, 1, entity.FieldTypeInt64, common.DefaultIntFieldName),
		common.GenColumnData(0, 1, entity.FieldTypeVarChar, common.DefaultVarcharFieldName),
		common.GenColumnData(0, 1, entity.FieldTypeFloatVector, common.DefaultFloatVecFieldName, common.WithVectorDim(common.DefaultDim)),
	}
	// invalid sparse vector: position > (maximum of uint32 - 1)
	positions = []uint32{math.MaxUint32}
	values = []float32{0.4}
	sparseVec, err := entity.NewSliceSparseEmbedding(positions, values)
	common.CheckErr(t, err, true)
	data1 := append(data, entity.NewColumnSparseVectors(common.DefaultSparseVecFieldName, []entity.SparseEmbedding{sparseVec}))
	_, err = mc.Insert(ctx, collName, "", data1...)
	common.CheckErr(t, err, false, "invalid index in sparse float vector: must be less than 2^32-1")
}

func TestInsertSparseVectorSamePosition(t *testing.T) {
	// invalid sparse vector: positions >= uint32
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	cp := CollectionParams{CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen}
	collName := createCollection(ctx, t, mc, cp)

	//insert data column
	data := []entity.Column{
		common.GenColumnData(0, 1, entity.FieldTypeInt64, common.DefaultIntFieldName),
		common.GenColumnData(0, 1, entity.FieldTypeVarChar, common.DefaultVarcharFieldName),
		common.GenColumnData(0, 1, entity.FieldTypeFloatVector, common.DefaultFloatVecFieldName, common.WithVectorDim(common.DefaultDim)),
	}
	//invalid sparse vector: position > (maximum of uint32 - 1)
	sparseVec, err := entity.NewSliceSparseEmbedding([]uint32{2, 10, 2}, []float32{0.4, 0.5, 0.6})
	common.CheckErr(t, err, true)
	data = append(data, entity.NewColumnSparseVectors(common.DefaultSparseVecFieldName, []entity.SparseEmbedding{sparseVec}))
	_, err = mc.Insert(ctx, collName, "", data...)
	common.CheckErr(t, err, false, "unsorted or same indices in sparse float vector")
}
