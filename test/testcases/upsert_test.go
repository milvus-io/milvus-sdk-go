//go:build L0

package testcases

import (
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestUpsert(t *testing.T) {
	/*
		1. prepare create -> insert -> index -> load -> query
		2. upsert exist entities -> data updated -> query and verify
		3. delete some pks -> query and verify
		4. upsert part deleted(not exist) pk and part existed pk -> query and verify
		5. upsert all not exist pk -> query and verify
	*/
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, enableDynamic := range []bool{true, false} {

		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := prepareCollection(ctx, t, mc, cp, WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

		upsertNb := 10
		// upsert exist entities [0, 10)
		data := common.GenAllFieldsData(0, upsertNb, common.DefaultDim)
		_, err := mc.Upsert(ctx, collName, "", data...)
		common.CheckErr(t, err, true)

		// query and verify the updated entities
		resSet, err := mc.Query(ctx, collName, []string{}, "int64 < 10", []string{common.DefaultFloatVecFieldName})
		common.CheckErr(t, err, true)
		idx := common.ColumnIndexFunc(data, common.DefaultFloatVecFieldName)
		require.ElementsMatch(t, data[idx].(*entity.ColumnFloatVector).Data()[:upsertNb],
			resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())

		// delete some pk
		mc.Delete(ctx, collName, "", "int64 < 10")
		resSet, err = mc.Query(ctx, collName, []string{}, "int64 < 10", []string{})
		require.Zero(t, resSet[0].Len())

		// upsert part deleted(not exist) pk and part existed pk [5, 15)
		data = common.GenAllFieldsData(5, upsertNb, common.DefaultDim)
		_, err = mc.Upsert(ctx, collName, "", data...)
		common.CheckErr(t, err, true)

		// query and verify the updated entities
		resSet, err = mc.Query(ctx, collName, []string{}, "5 <= int64 < 15", []string{common.DefaultFloatVecFieldName})
		common.CheckErr(t, err, true)
		require.ElementsMatch(t, data[idx].(*entity.ColumnFloatVector).Data()[:upsertNb],
			resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())

		// upsert all deleted(not exist) pk [0, 5)
		data = common.GenAllFieldsData(0, 5, common.DefaultDim)
		_, err = mc.Upsert(ctx, collName, "", data...)
		common.CheckErr(t, err, true)

		// query and verify the updated entities
		resSet, err = mc.Query(ctx, collName, []string{}, "0 <= int64 < 5", []string{common.DefaultFloatVecFieldName})
		common.CheckErr(t, err, true)
		require.ElementsMatch(t, data[idx].(*entity.ColumnFloatVector).Data()[:5],
			resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())
	}
}

// test upsert autoId collection
func TestUpsertAutoID(t *testing.T) {
	/*
		prepare autoID collection
		upsert not exist pk -> error
		upsert exist pk -> error ? autoID not supported upsert
	*/
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

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// index and load
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, "floatVec", idx, false)
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// upsert without pks
	_, floatColumn1, vecColumn1 := common.GenDefaultColumnData(0, 100, common.DefaultDim)
	_, err := mc.Upsert(ctx, collName, "", floatColumn1, vecColumn1)
	common.CheckErr(t, err, false, "upsert can not assign primary field data when auto id enabled")

	// upsert with pks
	pkColumn := entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:100])
	_, err = mc.Upsert(ctx, collName, "", pkColumn, floatColumn1, vecColumn1)
	common.CheckErr(t, err, false, "the number of fields is less than needed")
}

func TestUpsertVarcharPk(t *testing.T) {
	/*
		test upsert varchar pks
		upsert after query
		upsert "a" -> " a " -> actually new insert
	*/
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: VarcharBinaryVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 16)
	ip := []IndexParams{{BuildIndex: true, Index: idx, FieldName: common.DefaultBinaryVecFieldName, async: false}}
	collName := prepareCollection(ctx, t, mc, cp, WithIndexParams(ip), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	upsertNb := 10
	// upsert exist entities [0, 10) varchar: ["1", ... "9"]
	varcharColumn, binaryColumn := common.GenDefaultVarcharData(0, upsertNb, common.DefaultDim)
	_, err := mc.Upsert(ctx, collName, "", varcharColumn, binaryColumn)
	common.CheckErr(t, err, true)

	// query and verify the updated entities
	pkColumn := entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn.(*entity.ColumnVarChar).Data()[:upsertNb])
	resSet, err := mc.QueryByPks(ctx, collName, []string{}, pkColumn, []string{common.DefaultBinaryVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, binaryColumn.(*entity.ColumnBinaryVector).Data()[:upsertNb],
		resSet.GetColumn(common.DefaultBinaryVecFieldName).(*entity.ColumnBinaryVector).Data())

	// upsert varchar (with space): [" 1 ", ... " 9 "]
	varcharValues := make([]string, 0, upsertNb)
	for i := 0; i < upsertNb; i++ {
		varcharValues = append(varcharValues, " "+strconv.Itoa(i)+" ")
	}
	varcharColumn1 := entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharValues)
	_, binaryColumn1 := common.GenDefaultVarcharData(0, upsertNb, common.DefaultDim)
	ids, err := mc.Upsert(ctx, collName, "", varcharColumn1, binaryColumn1)
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, ids.(*entity.ColumnVarChar).Data(), varcharValues)

	// query old varchar pk (no space): ["1", ... "9"]
	resSet, err = mc.QueryByPks(ctx, collName, []string{}, pkColumn, []string{common.DefaultVarcharFieldName, common.DefaultBinaryVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, varcharColumn.(*entity.ColumnVarChar).Data()[:upsertNb], resSet.GetColumn(common.DefaultVarcharFieldName).(*entity.ColumnVarChar).Data())
	require.ElementsMatch(t, binaryColumn.(*entity.ColumnBinaryVector).Data()[:upsertNb], resSet.GetColumn(common.DefaultBinaryVecFieldName).(*entity.ColumnBinaryVector).Data())

	// query and verify the updated entities
	pkColumn1 := entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn1.Data())
	resSet, err = mc.QueryByPks(ctx, collName, []string{}, pkColumn1, []string{common.DefaultVarcharFieldName, common.DefaultBinaryVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, varcharColumn1.Data(), resSet.GetColumn(common.DefaultVarcharFieldName).(*entity.ColumnVarChar).Data())
	require.ElementsMatch(t, binaryColumn1.(*entity.ColumnBinaryVector).Data(), resSet.GetColumn(common.DefaultBinaryVecFieldName).(*entity.ColumnBinaryVector).Data())
}

// test upsert with partition
func TestUpsertMultiPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards, client.WithConsistencyLevel(entity.ClStrong))

	// insert [0, nb) into default, insert [nb, nb*2) into new
	_, defaultPartition, newPartition := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// upsert new partition
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, 100, common.DefaultDim)
	_, err := mc.Upsert(ctx, collName, newPartition.PartitionName, intColumn, floatColumn, vecColumn)
	common.CheckErr(t, err, true)

	// query and verify
	resSet, err := mc.QueryByPks(ctx, collName, []string{newPartition.PartitionName}, intColumn,
		[]string{common.DefaultFloatVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, vecColumn.(*entity.ColumnFloatVector).Data()[:100], resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())

	// query and verify
	resSet, err = mc.QueryByPks(ctx, collName, []string{defaultPartition.PartitionName}, intColumn,
		[]string{common.DefaultFloatVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, defaultPartition.VectorColumn.(*entity.ColumnFloatVector).Data()[:100], resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())
}

// test upsert with invalid collection / partition name
func TestUpsertNotExistCollectionPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect
	mc := createMilvusClient(ctx, t)

	// create default collection with autoID true
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)

	// upsert not exist partition
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, errUpsert := mc.Upsert(ctx, collName, "aaa", floatColumn, vecColumn)
	common.CheckErr(t, errUpsert, false, "does not exist")

	// upsert not exist collection
	_, errUpsert = mc.Upsert(ctx, "aaa", "", floatColumn, vecColumn)
	common.CheckErr(t, errUpsert, false, "does not exist")
}

// test upsert with invalid column data
func TestUpsertInvalidColumnData(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: Int64FloatVecJSON, start: 0, nb: 200,
		dim: common.DefaultDim, EnableDynamicField: false}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp),
		WithIndexParams([]IndexParams{{BuildIndex: false}}),
		WithLoadParams(LoadParams{DoLoad: false}), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	upsertNb := 10
	// 1. upsert missing columns
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, upsertNb, common.DefaultDim)
	jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, upsertNb)
	_, err := mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, err, false, fmt.Sprintf("field %s not passed", common.DefaultJSONFieldName))

	// 2. upsert extra a column
	varColumn := common.GenColumnData(0, upsertNb, entity.FieldTypeVarChar, common.DefaultVarcharFieldName)
	_, err = mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn, jsonColumn, varColumn)
	common.CheckErr(t, err, false, fmt.Sprintf("field %s does not exist in collection", common.DefaultVarcharFieldName))

	// 3. upsert vector has different dim
	dimColumn := common.GenColumnData(0, upsertNb, entity.FieldTypeFloatVector, common.DefaultFloatVecFieldName, common.WithVectorDim(64))
	_, err = mc.Upsert(ctx, collName, "", intColumn, floatColumn, dimColumn, jsonColumn)
	common.CheckErr(t, err, false, fmt.Sprintf("params column %s vector dim 64 not match collection definition, which has dim of %d",
		common.DefaultFloatVecFieldName, common.DefaultDim))

	// 4. different columns has different length
	diffLenColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, upsertNb*2)
	_, err = mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn, diffLenColumn)
	common.CheckErr(t, err, false, "column size not match")

	// 5. column type different with schema
	_, err = mc.Upsert(ctx, collName, "", intColumn, varColumn, vecColumn, jsonColumn)
	common.CheckErr(t, err, false, "field varchar does not exist in collection")
}

func TestUpsertSamePksManyTimes(t *testing.T) {
	// upsert pks [0, 1000) many times with different vector
	// I mean many delete + insert
	// query -> gets last upsert entities
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := prepareCollection(ctx, t, mc, cp, WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	var data []entity.Column
	upsertNb := 1000
	for i := 0; i < 100; i++ {
		// upsert exist entities [0, 10)
		data = common.GenAllFieldsData(0, upsertNb, common.DefaultDim)
		_, err := mc.Upsert(ctx, collName, "", data...)
		common.CheckErr(t, err, true)
	}

	// query and verify the updated entities
	resSet, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < %d", common.DefaultIntFieldName, upsertNb), []string{common.DefaultFloatVecFieldName})
	common.CheckErr(t, err, true)
	idx := common.ColumnIndexFunc(data, common.DefaultFloatVecFieldName)
	require.ElementsMatch(t, data[idx].(*entity.ColumnFloatVector).Data(),
		resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())
}

func TestUpsertDynamicField(t *testing.T) {
	// enable dynamic field and insert dynamic column
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index -> load
	cp := CollectionParams{CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: Int64FloatVec, start: 0, nb: common.DefaultNb,
		dim: common.DefaultDim, EnableDynamicField: true}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// verify that dynamic field exists
	upsertNb := 10
	resSet, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < %d", common.DefaultDynamicNumberField, upsertNb),
		[]string{common.DefaultDynamicNumberField})
	require.Equal(t, upsertNb, resSet[0].Len())

	// 1. upsert exist pk without dynamic column
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, upsertNb, common.DefaultDim)
	_, err = mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn)
	common.CheckErr(t, err, true)

	// query and gets empty
	resSet, err = mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < %d", common.DefaultDynamicNumberField, upsertNb), []string{common.DefaultDynamicNumberField})
	require.Zero(t, resSet[0].Len())

	// 2. upsert not exist pk with dynamic column ->  field dynamicNumber does not exist in collection
	intColumn2, floatColumn2, vecColumn2 := common.GenDefaultColumnData(common.DefaultNb, upsertNb, common.DefaultDim)
	dynamicData2 := common.GenDynamicFieldData(common.DefaultNb, upsertNb)
	_, err = mc.Upsert(ctx, collName, "", append(dynamicData2, intColumn2, floatColumn2, vecColumn2)...)
	common.CheckErr(t, err, true)

	// query and gets empty dynamic field
	resSet, err = mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s >= %d", common.DefaultDynamicNumberField, common.DefaultNb), []string{common.QueryCountFieldName})
	require.Equal(t, int64(upsertNb), resSet.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
}

func TestUpsertPartitionKeyCollection(t *testing.T) {
	// upsert data into collection that has partition key field
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create partition_key field
	partitionKeyFieldName := "partitionKeyField"
	partitionKeyField := common.GenField(partitionKeyFieldName, entity.FieldTypeInt64,
		common.WithIsPartitionKey(true), common.WithMaxLength(common.TestMaxLen))

	// schema
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false))
	schema.WithField(partitionKeyField)

	// create collection and check partition key
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10),
		client.WithConsistencyLevel(entity.ClStrong))
	common.CheckErr(t, err, true)

	// insert data partition key field [0, nb)
	partitionKeyColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeInt64, partitionKeyFieldName)
	pkColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, schema.CollectionName, "", pkColumn, floatColumn, vecColumn, partitionKeyColumn)
	common.CheckErr(t, errInsert, true)
	require.Equalf(t, common.DefaultNb, ids.Len(), fmt.Sprintf("Expected insert result equal to %d, actual %d", common.DefaultNb, ids.Len()))

	// flush -> index and load
	mc.Flush(ctx, schema.CollectionName, false)
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, schema.CollectionName, common.DefaultFloatVecFieldName, idx, false)
	mc.LoadCollection(ctx, schema.CollectionName, false)

	// upsert data partition key field [nb, nb*2)
	partitionKeyColumn2 := common.GenColumnData(common.DefaultNb, common.DefaultNb, entity.FieldTypeInt64, partitionKeyFieldName)
	mc.Upsert(ctx, schema.CollectionName, "", pkColumn, floatColumn, vecColumn, partitionKeyColumn2)

	// verify upsert
	resSet, err := mc.Query(ctx, schema.CollectionName, []string{}, fmt.Sprintf("%s >= %d", partitionKeyFieldName, common.DefaultNb),
		[]string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb), resSet.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	resSet, err = mc.Query(ctx, schema.CollectionName, []string{}, fmt.Sprintf("%s < %d", partitionKeyFieldName, common.DefaultNb),
		[]string{common.QueryCountFieldName})
	require.Equal(t, int64(0), resSet.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
}

func TestUpsertWithoutLoading(t *testing.T) {
	// test upsert without loading (because delete need loading)
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create -> insert [0, 3000) -> flush -> index
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}

	dp := DataParams{DoInsert: true, CollectionFieldsType: Int64FloatVecJSON, start: 0, nb: 200,
		dim: common.DefaultDim, EnableDynamicField: true}
	collName := prepareCollection(ctx, t, mc, cp, WithDataParams(dp),
		WithFlushParams(FlushParams{DoFlush: false}),
		WithIndexParams([]IndexParams{{BuildIndex: false}}),
		WithLoadParams(LoadParams{DoLoad: false}), WithCreateOption(client.WithConsistencyLevel(entity.ClStrong)))

	// upsert
	upsertNb := 10
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, upsertNb, common.DefaultDim)
	jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, upsertNb)
	_, err := mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn, jsonColumn)
	common.CheckErr(t, err, true)

	// index -> load
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)
	mc.LoadCollection(ctx, collName, false)
	err = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// query and verify
	resSet, err := mc.QueryByPks(ctx, collName, []string{}, intColumn,
		[]string{common.DefaultFloatVecFieldName})
	common.CheckErr(t, err, true)
	require.ElementsMatch(t, vecColumn.(*entity.ColumnFloatVector).Data()[:upsertNb], resSet.GetColumn(common.DefaultFloatVecFieldName).(*entity.ColumnFloatVector).Data())
}
