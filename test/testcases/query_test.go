//go:build L0

package testcases

import (
	"encoding/json"
	"fmt"
	"log"
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
	var queryResult, _ = mc.QueryByPks(
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
	queryResult, _ := mc.QueryByPks(
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
	_, errQuery := mc.QueryByPks(
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
	_, errQuery := mc.QueryByPks(
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
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

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
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{emptyPartitionName},
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.(*entity.ColumnInt64).Data()[:10]),
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, false, "Partition name should not be empty")
}

// query with empty partition names, actually query from all partitions
func TestQueryMultiPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert [0, nb) into default partition, [nb, nb*2) into new partition
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
	partitionName, _, _ := createInsertTwoPartitions(ctx, t, mc, collName, common.DefaultNb)

	// create index
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query from multi partition names
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, common.DefaultNb, common.DefaultNb*2 - 1})
	queryResultMultiPartition, _ := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition, partitionName}, queryIds,
		[]string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{queryIds})

	//query from empty partition names, expect to query from all partitions
	queryResultEmptyPartition, _ := mc.QueryByPks(ctx, collName, []string{}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultEmptyPartition, []entity.Column{queryIds})

	// query from new partition and query successfully
	queryResultPartition, _ := mc.QueryByPks(ctx, collName, []string{partitionName}, queryIds, []string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultPartition,
		[]entity.Column{entity.NewColumnInt64(common.DefaultIntFieldName, []int64{common.DefaultNb, common.DefaultNb*2 - 1})})

	// query from new partition and query gets empty pk column data
	queryResultEmpty, errQuery := mc.QueryByPks(ctx, collName, []string{partitionName}, entity.NewColumnInt64(common.DefaultIntFieldName,
		[]int64{0}), []string{common.DefaultIntFieldName})
	common.CheckErr(t, errQuery, true)
	require.Equalf(t, queryResultEmpty[0].Len(), 0,
		fmt.Sprintf("Expected query return empty pk column data, but gets data: %d", queryResultEmpty[0].(*entity.ColumnInt64).Data()))
}

// test query with empty ids
// TODO Issue: https://github.com/milvus-io/milvus-sdk-go/issues/365
func TestQueryEmptyIds(t *testing.T) {
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
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		queryIds,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckErr(t, errQuery, false, "ids len must not be zero")
}

// test query with non-primary field filter, and output scalar fields
func TestQueryNonPrimaryFields(t *testing.T) {
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
		_, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition}, idsColumn,
			[]string{common.DefaultIntFieldName})

		// TODO only int64 and varchar column can be primary key for now
		common.CheckErr(t, errQuery, false, "only int64 and varchar column can be primary key for now")
		//common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0})})
	}
}

// test query empty or one scalar output fields
func TestQueryEmptyOutputFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, enableDynamic := range []bool{true, false} {
		// create, insert, index
		collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true, client.WithEnableDynamicSchema(enableDynamic))

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		//query with empty output fields []string{}-> output "int64"
		queryEmptyOutputs, _ := mc.QueryByPks(
			ctx, collName, []string{common.DefaultPartition},
			entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
			[]string{},
		)
		common.CheckOutputFields(t, queryEmptyOutputs, []string{common.DefaultIntFieldName})

		//query with empty output fields []string{""}-> output "int64" and dynamic field
		queryEmptyOutputs, err := mc.QueryByPks(
			ctx, collName, []string{common.DefaultPartition},
			entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
			[]string{""},
		)
		if enableDynamic {
			common.CheckErr(t, err, false, "parse output field name failed")
		} else {
			common.CheckErr(t, err, false, "not exist")
		}

		// query with "float" output fields -> output "int64, float"
		queryFloatOutputs, _ := mc.QueryByPks(
			ctx, collName, []string{common.DefaultPartition},
			entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
			[]string{common.DefaultFloatFieldName},
		)
		common.CheckOutputFields(t, queryFloatOutputs, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName})
	}
}

// test query output int64 and float and floatVector fields
func TestQueryOutputFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection and insert data into default partition, pks from 0 to DefaultNb
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)
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
	queryResult, _ := mc.QueryByPks(
		ctx, collName,
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.(*entity.ColumnInt64).Data()[:pos]),
		[]string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnInt64(common.DefaultIntFieldName, intColumn.(*entity.ColumnInt64).Data()[:pos]),
		entity.NewColumnFloat(common.DefaultFloatFieldName, floatColumn.(*entity.ColumnFloat).Data()[:pos]),
		entity.NewColumnFloatVector(common.DefaultFloatVecFieldName, int(common.DefaultDim), vecColumn.(*entity.ColumnFloatVector).Data()[:pos]),
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
	queryResult, _ := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn.(*entity.ColumnVarChar).Data()[:pos]),
		[]string{common.DefaultBinaryVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnVarChar(common.DefaultVarcharFieldName, varcharColumn.(*entity.ColumnVarChar).Data()[:pos]),
		entity.NewColumnBinaryVector(common.DefaultBinaryVecFieldName, int(common.DefaultDim), vecColumn.(*entity.ColumnBinaryVector).Data()[:pos]),
	})
	common.CheckOutputFields(t, queryResult, []string{common.DefaultBinaryVecFieldName, common.DefaultVarcharFieldName})
}

// test query output all fields
func TestOutputAllFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, withRows := range []bool{true, false} {
		// create collection
		var capacity int64 = common.TestCapacity
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity}
		collName := createCollection(ctx, t, mc, cp)

		// prepare and insert data
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows}
		_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))

		// flush and check row count
		errFlush := mc.Flush(ctx, collName, false)
		common.CheckErr(t, errFlush, true)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		for _, fieldName := range []string{"floatVec", "fp16Vec", "bf16Vec", "binaryVec"} {
			_ = mc.CreateIndex(ctx, collName, fieldName, idx, false)
		}

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// query output all fields -> output all fields, includes vector and $meta field
		allFieldsName := append(common.AllArrayFieldsName, "int64", "bool", "int8", "int16", "int32", "float",
			"double", "varchar", "json", "floatVec", "fp16Vec", "bf16Vec", "binaryVec", common.DefaultDynamicFieldName)
		queryResultAll, errQuery := mc.Query(ctx, collName, []string{},
			fmt.Sprintf("%s == 0", common.DefaultIntFieldName), []string{"*"})
		common.CheckErr(t, errQuery, true)
		common.CheckOutputFields(t, queryResultAll, allFieldsName)
	}
}

// test query with an not existed field
func TestQueryOutputNotExistField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	//query
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		entity.NewColumnInt64(common.DefaultIntFieldName, ids.(*entity.ColumnInt64).Data()[:10]),
		[]string{common.DefaultIntFieldName, "varchar"},
	)
	common.CheckErr(t, errQuery, false, "field varchar not exist")
}

// Test query json collection, filter json field, output json field
func TestQueryJsonDynamicField(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, dynamicField := range []bool{true, false} {
		// create collection
		cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: dynamicField,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: dynamicField}
		_, _ = insertData(ctx, t, mc, dp)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		outputFields := []string{common.DefaultIntFieldName, common.DefaultJSONFieldName}
		if dynamicField {
			outputFields = append(outputFields, common.DefaultDynamicFieldName)
		}

		// query and output json/dynamic field
		pkColumn := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1})
		queryResult, err := mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			pkColumn,
			outputFields,
		)
		common.CheckErr(t, err, true)
		m0 := common.JSONStruct{String: strconv.Itoa(0), Bool: true}
		j0, _ := json.Marshal(&m0)
		m1 := common.JSONStruct{Number: int32(1), String: strconv.Itoa(1), Bool: false, List: []int64{int64(1), int64(2)}}
		j1, _ := json.Marshal(&m1)
		jsonValues := [][]byte{j0, j1}
		jsonColumn := entity.NewColumnJSONBytes(common.DefaultJSONFieldName, jsonValues)
		actualColumns := []entity.Column{pkColumn, jsonColumn}
		if dynamicField {
			dynamicColumn := common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2))
			actualColumns = append(actualColumns, dynamicColumn)
		}

		for _, column := range queryResult {
			log.Println(column.FieldData())
			if column.Type() == entity.FieldTypeJSON {
				var jsonData []string
				for i := 0; i < column.Len(); i++ {
					line, err := column.GetAsString(i)
					if err != nil {
						fmt.Println(err)
					}
					jsonData = append(jsonData, line)
				}
				log.Println(jsonData)
			}
		}
		common.CheckQueryResult(t, queryResult, actualColumns)

		// query with json column
		queryResult, err = mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			jsonColumn,
			[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName},
		)
		common.CheckErr(t, err, false, "only int64 and varchar column can be primary key for now")

		// query with dynamic column
		queryResult, err = mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2)),
			[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName},
		)
		common.CheckErr(t, err, false, "only int64 and varchar column can be primary key for now")
	}
}

// Test query json and dynamic collection with string expr
func TestQueryCountJsonDynamicExpr(t *testing.T) {
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

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query with different expr and count
	type exprCount struct {
		expr  string
		count int64
	}
	exprCounts := []exprCount{
		{expr: "", count: common.DefaultNb},
		// pk int64 field expr: < in && ||
		{expr: fmt.Sprintf("%s < 1000", common.DefaultIntFieldName), count: 1000},
		{expr: fmt.Sprintf("%s in [0, 1, 2]", common.DefaultIntFieldName), count: 3},
		{expr: fmt.Sprintf("%s >= 1000 && %s < 2000", common.DefaultIntFieldName, common.DefaultIntFieldName), count: 1000},
		{expr: fmt.Sprintf("%s >= 1000 || %s > 2000", common.DefaultIntFieldName, common.DefaultIntFieldName), count: 2000},
		{expr: fmt.Sprintf("%s < 1000", common.DefaultFloatFieldName), count: 1000},

		// json and dynamic field filter expr: == < in bool/ list/ int
		{expr: fmt.Sprintf("%s['number'] == 0", common.DefaultJSONFieldName), count: 1500 / 2},
		{expr: fmt.Sprintf("%s['number'] < 100 and %s['number'] != 0", common.DefaultJSONFieldName, common.DefaultJSONFieldName), count: 50},
		{expr: fmt.Sprintf("%s < 100", common.DefaultDynamicNumberField), count: 100},
		{expr: "dynamicNumber % 2 == 0", count: 1500},
		{expr: fmt.Sprintf("%s['bool'] == true", common.DefaultJSONFieldName), count: 1500 / 2},
		{expr: fmt.Sprintf("%s == false", common.DefaultDynamicBoolField), count: 2000},
		{expr: fmt.Sprintf("%s in ['1', '2'] ", common.DefaultDynamicStringField), count: 2},
		{expr: fmt.Sprintf("%s['string'] in ['1', '2', '5'] ", common.DefaultJSONFieldName), count: 3},
		{expr: fmt.Sprintf("%s['list'] == [1, 2] ", common.DefaultJSONFieldName), count: 1},
		{expr: fmt.Sprintf("%s['list'] == [0, 1] ", common.DefaultJSONFieldName), count: 0},
		{expr: fmt.Sprintf("%s['list'][0] < 10 ", common.DefaultJSONFieldName), count: 5},
		{expr: fmt.Sprintf("%s[\"dynamicList\"] != [2, 3]", common.DefaultDynamicFieldName), count: 0},

		// json contains
		{expr: fmt.Sprintf("json_contains (%s['list'], 2)", common.DefaultJSONFieldName), count: 1},
		{expr: fmt.Sprintf("json_contains (%s['number'], 0)", common.DefaultJSONFieldName), count: 0},
		{expr: fmt.Sprintf("json_contains_all (%s['list'], [1, 2])", common.DefaultJSONFieldName), count: 1},
		{expr: fmt.Sprintf("JSON_CONTAINS_ANY (%s['list'], [1, 3])", common.DefaultJSONFieldName), count: 2},
		// string like
		{expr: "dynamicString like '1%' ", count: 1111},

		// key exist
		{expr: fmt.Sprintf("exists %s['list']", common.DefaultJSONFieldName), count: common.DefaultNb / 2},
		{expr: fmt.Sprintf("exists a "), count: 0},
		{expr: fmt.Sprintf("exists %s ", common.DefaultDynamicListField), count: 0},
		{expr: fmt.Sprintf("exists %s ", common.DefaultDynamicStringField), count: common.DefaultNb},
		// data type not match and no error
		{expr: fmt.Sprintf("%s['number'] == '0' ", common.DefaultJSONFieldName), count: 0},

		// json field
		{expr: fmt.Sprintf("%s >= 1500", common.DefaultJSONFieldName), count: 1500 / 2},    // json >= 1500
		{expr: fmt.Sprintf("%s > 1499.5", common.DefaultJSONFieldName), count: 1500 / 2},   // json >= 1500.0
		{expr: fmt.Sprintf("%s like '21%%'", common.DefaultJSONFieldName), count: 100 / 4}, // json like '21%'
		{expr: fmt.Sprintf("%s == [1503, 1504]", common.DefaultJSONFieldName), count: 1},   // json == [1,2]
		{expr: fmt.Sprintf("%s[0] > 1", common.DefaultJSONFieldName), count: 1500 / 4},     // json[0] > 1
		{expr: fmt.Sprintf("%s[0][0] > 1", common.DefaultJSONFieldName), count: 0},         // json == [1,2]
	}

	for _, _exprCount := range exprCounts {
		countRes, _ := mc.Query(ctx, collName,
			[]string{common.DefaultPartition},
			_exprCount.expr, []string{common.QueryCountFieldName})
		require.Equal(t, _exprCount.count, countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
	}
}

// test query with all kinds of array expr
func TestQueryArrayFieldExpr(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, withRows := range []bool{true, false} {
		// create collection
		var capacity int64 = common.TestCapacity
		cp := CollectionParams{CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity}
		collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

		// prepare and insert data
		dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows}
		_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))

		// flush and check row count
		errFlush := mc.Flush(ctx, collName, false)
		common.CheckErr(t, errFlush, true)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		type exprCount struct {
			expr  string
			count int64
		}
		exprCounts := []exprCount{
			{expr: fmt.Sprintf("%s[0] == false", common.DefaultBoolArrayField), count: common.DefaultNb / 2},                 //  array[0] ==
			{expr: fmt.Sprintf("%s[0] > 0", common.DefaultInt64ArrayField), count: common.DefaultNb - 1},                     //  array[0] >
			{expr: fmt.Sprintf("%s[0] > 0", common.DefaultInt8ArrayField), count: 1524},                                      //  array[0] > int8 range: [-128, 127]
			{expr: fmt.Sprintf("json_contains (%s, %d)", common.DefaultInt16ArrayField, capacity), count: capacity},          // json_contains(array, 1)
			{expr: fmt.Sprintf("array_contains (%s, %d)", common.DefaultInt16ArrayField, capacity), count: capacity},         // array_contains(array, 1)
			{expr: fmt.Sprintf("array_contains (%s, 1)", common.DefaultInt32ArrayField), count: 2},                           // array_contains(array, 1)
			{expr: fmt.Sprintf("json_contains (%s, 1)", common.DefaultInt32ArrayField), count: 2},                            // json_contains(array, 1)
			{expr: fmt.Sprintf("array_contains (%s, 1000000)", common.DefaultInt32ArrayField), count: 0},                     // array_contains(array, 1)
			{expr: fmt.Sprintf("json_contains_all (%s, [90, 91])", common.DefaultInt64ArrayField), count: 91},                // json_contains_all(array, [x])
			{expr: fmt.Sprintf("array_contains_all (%s, [1, 2])", common.DefaultInt64ArrayField), count: 2},                  // array_contains_all(array, [x])
			{expr: fmt.Sprintf("array_contains_any (%s, [0, 100, 10000])", common.DefaultFloatArrayField), count: 101},       // array_contains_any(array, [x])
			{expr: fmt.Sprintf("json_contains_any (%s, [0, 100, 10])", common.DefaultFloatArrayField), count: 101},           // json_contains_any (array, [x])
			{expr: fmt.Sprintf("%s == [0, 1]", common.DefaultDoubleArrayField), count: 0},                                    //  array ==
			{expr: fmt.Sprintf("array_length(%s) == 10", common.DefaultVarcharArrayField), count: 0},                         //  array_length
			{expr: fmt.Sprintf("array_length(%s) == %d", common.DefaultDoubleArrayField, capacity), count: common.DefaultNb}, //  array_length
		}

		for _, _exprCount := range exprCounts {
			log.Println(_exprCount.expr)
			countRes, err := mc.Query(ctx, collName,
				[]string{}, _exprCount.expr, []string{common.QueryCountFieldName})
			common.CheckErr(t, err, true)
			require.Equal(t, _exprCount.count, countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
		}
	}
}

// test query different array rows has different element length
func TestQueryArrayDifferentLenBetweenRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// fields
	defaultFields := common.GenDefaultFields(false)
	int64ArrayField := common.GenField(common.DefaultInt32ArrayField, entity.FieldTypeArray,
		common.WithElementType(entity.FieldTypeInt32), common.WithMaxCapacity(common.TestCapacity))

	// create collection with max
	collName := common.GenRandomString(4)
	schema := common.GenSchema(collName, false, append(defaultFields, int64ArrayField), common.WithEnableDynamicField(true))
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithConsistencyLevel(entity.ClStrong))
	common.CheckErr(t, err, true)

	// insert data [0, 1500) with array length = capacity, values: [i+0, i+ 100)
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb/2, common.DefaultDim)
	data := []entity.Column{
		intColumn,
		floatColumn,
		vecColumn,
		common.GenArrayColumnData(0, common.DefaultNb/2, common.DefaultInt32ArrayField,
			common.WithArrayElementType(entity.FieldTypeInt32), common.WithArrayCapacity(common.TestCapacity)),
	}
	ids, err := mc.Insert(ctx, collName, "", data...)
	common.CheckErr(t, err, true)

	// insert data [1500, 3000) with array length = capacity/2, values: values: [i+0, i+ 50)
	require.Equal(t, common.DefaultNb/2, ids.Len())
	intColumn1, floatColumn1, vecColumn1 := common.GenDefaultColumnData(common.DefaultNb/2, common.DefaultNb/2, common.DefaultDim)
	data1 := []entity.Column{
		intColumn1,
		floatColumn1,
		vecColumn1,
		common.GenArrayColumnData(common.DefaultNb/2, common.DefaultNb/2, common.DefaultInt32ArrayField,
			common.WithArrayElementType(entity.FieldTypeInt32), common.WithArrayCapacity(common.TestCapacity/2)),
	}
	ids, err = mc.Insert(ctx, collName, "", data1...)
	common.CheckErr(t, err, true)
	require.Equal(t, common.DefaultNb/2, ids.Len())

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query array idx exceeds max capacity, array[100]
	countRes, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s[%d] > 0", common.DefaultInt32ArrayField, common.TestCapacity),
		[]string{common.QueryCountFieldName})
	common.CheckErr(t, err, true)
	require.Equal(t, int64(0), countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// query: some rows has element less than expr index array[51]
	countRes, err = mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s[%d] > 0", common.DefaultInt32ArrayField, common.TestCapacity/2+1),
		[]string{common.QueryCountFieldName})
	common.CheckErr(t, err, true)
	require.Equal(t, int64(common.DefaultNb/2), countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
}

// test query with expr and verify output dynamic field data
func TestQueryJsonDynamicExpr(t *testing.T) {
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

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query with different expr and count
	expr := fmt.Sprintf("%s['number'] < 10 and %s < 10", common.DefaultJSONFieldName, common.DefaultDynamicNumberField)
	queryRes, err := mc.Query(ctx, collName,
		[]string{common.DefaultPartition},
		expr, []string{common.DefaultJSONFieldName, common.DefaultDynamicNumberField})

	if err != nil {
		log.Println(err)
	}
	// verify output fields and count, dynamicNumber value
	common.CheckOutputFields(t, queryRes, []string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicNumberField})
	require.Equal(t, 10, queryRes.GetColumn(common.DefaultJSONFieldName).Len())
	dynamicNumColumn := queryRes.GetColumn(common.DefaultDynamicNumberField)
	var numberData []int64
	for i := 0; i < dynamicNumColumn.Len(); i++ {
		line, _ := dynamicNumColumn.GetAsInt64(i)
		numberData = append(numberData, line)
	}
	require.Equal(t, numberData, []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
}

// test query and output both json and dynamic field
func TestQueryJsonDynamicFieldRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: true}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	countRes, _ := mc.Query(ctx, collName,
		[]string{common.DefaultPartition},
		"", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb), countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// query and output json field
	pkColumn := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1})
	queryResult, err := mc.QueryByPks(
		ctx, collName,
		[]string{common.DefaultPartition},
		pkColumn,
		[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicFieldName},
	)
	common.CheckErr(t, err, true)
	//jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, 2)
	m0 := common.JSONStruct{String: strconv.Itoa(0), Bool: true}
	j0, _ := json.Marshal(&m0)
	m1 := common.JSONStruct{Number: int32(1), String: strconv.Itoa(1), Bool: false, List: []int64{int64(1), int64(2)}}
	j1, _ := json.Marshal(&m1)
	jsonValues := [][]byte{j0, j1}
	jsonColumn := entity.NewColumnJSONBytes(common.DefaultJSONFieldName, jsonValues)
	dynamicColumn := common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2))
	// gen dynamic json column

	for _, column := range queryResult {
		log.Printf("name: %s, type: %s, fieldData: %v", column.Name(), column.Type(), column.FieldData())
		if column.Type() == entity.FieldTypeJSON {
			var jsonData []string
			for i := 0; i < column.Len(); i++ {
				line, err := column.GetAsString(i)
				if err != nil {
					fmt.Println(err)
				}
				jsonData = append(jsonData, line)
			}
			log.Println(jsonData)
		}
	}
	common.CheckQueryResult(t, queryResult, []entity.Column{pkColumn, jsonColumn, dynamicColumn})

	// query with different expr and count
	expr := fmt.Sprintf("%s['number'] < 10 && %s < 10", common.DefaultJSONFieldName, common.DefaultDynamicNumberField)
	queryRes, _ := mc.Query(ctx, collName,
		[]string{common.DefaultPartition},
		expr, []string{common.DefaultJSONFieldName, common.DefaultDynamicNumberField})

	// verify output fields and count, dynamicNumber value
	common.CheckOutputFields(t, queryRes, []string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicNumberField})
	require.Equal(t, 10, queryRes.GetColumn(common.DefaultJSONFieldName).Len())
	dynamicNumColumn := queryRes.GetColumn(common.DefaultDynamicNumberField)
	var numberData []int64
	for i := 0; i < dynamicNumColumn.Len(); i++ {
		line, _ := dynamicNumColumn.GetAsInt64(i)
		numberData = append(numberData, line)
	}
	require.Equal(t, numberData, []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
}

// test query with invalid expr
func TestQueryInvalidExpr(t *testing.T) {
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

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	for _, _invalidExprs := range common.InvalidExpressions {
		_, err := mc.Query(ctx, collName,
			[]string{common.DefaultPartition},
			_invalidExprs.Expr, []string{common.DefaultJSONFieldName, common.DefaultDynamicNumberField})
		common.CheckErr(t, err, _invalidExprs.ErrNil, _invalidExprs.ErrMsg)
	}
}

// test query output invalid count(*) fields
func TestQueryOutputInvalidOutputFieldCount(t *testing.T) {
	type invalidCountStruct struct {
		countField string
		errMsg     string
	}
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create, insert, index
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true,
		client.WithEnableDynamicSchema(false), client.WithConsistencyLevel(entity.ClStrong))

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// invalid expr
	invalidOutputFieldCount := []invalidCountStruct{
		{countField: "Count(*)", errMsg: "extra output fields [Count(*)] found and result does not dynamic field"},
		{countField: "ccount(*)", errMsg: "field ccount(*) not exist"},
		{countField: "count[*]", errMsg: "field count[*] not exist"},
		{countField: "count", errMsg: "field count not exist"},
		{countField: "count(**)", errMsg: "field count(**) not exist"},
	}
	for _, invalidCount := range invalidOutputFieldCount {
		queryExpr := fmt.Sprintf("%s >= 0", common.DefaultIntFieldName)

		//query with empty output fields []string{}-> output "int64"
		_, err := mc.Query(
			ctx, collName, []string{common.DefaultPartition},
			queryExpr, []string{invalidCount.countField})
		common.CheckErr(t, err, false, invalidCount.errMsg)
	}
}

// test query count* after insert -> delete -> upsert -> compact
func TestQueryCountAfterDml(t *testing.T) {
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
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

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

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// count*
	countQuery, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb), countQuery.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	//inert 1000 entities -> count*
	insertNb := 1000
	dpInsert := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: common.DefaultNb, nb: insertNb, dim: common.DefaultDim, EnableDynamicField: true}
	insertData(ctx, t, mc, dpInsert)
	countAfterInsert, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb+insertNb), countAfterInsert.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// delete 1000 entities -> count*
	mc.Delete(ctx, collName, common.DefaultPartition, fmt.Sprintf("%s < 1000 ", common.DefaultIntFieldName))
	countAfterDelete, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb), countAfterDelete.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// upsert deleted 100 entities -> count*
	upsertNb := 100
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, upsertNb, common.DefaultDim)
	jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, upsertNb)
	mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn, jsonColumn)
	countAfterUpsert, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb+upsertNb), countAfterUpsert.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// upsert existed 100 entities -> count*
	intColumn, floatColumn, vecColumn = common.GenDefaultColumnData(common.DefaultNb, upsertNb, common.DefaultDim)
	jsonColumn = common.GenDefaultJSONData(common.DefaultJSONFieldName, common.DefaultNb, upsertNb)
	mc.Upsert(ctx, collName, "", intColumn, floatColumn, vecColumn, jsonColumn)
	countAfterUpsert2, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb+upsertNb), countAfterUpsert2.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

	// compact -> count(*)
	_, err := mc.Compact(ctx, collName, time.Second*60)
	common.CheckErr(t, err, true)
	countAfterCompact, _ := mc.Query(ctx, collName, []string{common.DefaultPartition}, "", []string{common.QueryCountFieldName})
	require.Equal(t, int64(common.DefaultNb+upsertNb), countAfterCompact.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
}

// TODO offset and limit
// TODO consistency level
// TODO ignore growing
