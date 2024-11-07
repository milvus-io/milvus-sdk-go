//go:build L0

package testcases

import (
	"encoding/json"
	"fmt"
	"io"
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

	// query
	pks := ids.Slice(0, 10)
	queryResult, _ := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		pks,
		[]string{common.DefaultIntFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{pks})
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

	// query
	pks := ids.Slice(0, 10)
	queryResult, _ := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		pks,
		[]string{common.DefaultVarcharFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{pks})
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

	// query
	pks := ids.Slice(0, 10)
	_, errQuery := mc.QueryByPks(
		ctx,
		"collName",
		[]string{common.DefaultPartition},
		pks,
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

	// query
	pks := ids.Slice(0, 10)
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{"aaa"},
		pks,
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

	// query from "" partitions, expect to query from default partition
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{emptyPartitionName},
		intColumn.Slice(0, 10),
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

	// query from multi partition names
	queryIds := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, common.DefaultNb, common.DefaultNb*2 - 1})
	queryResultMultiPartition, _ := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition, partitionName}, queryIds,
		[]string{common.DefaultIntFieldName})
	common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{queryIds})

	// query from empty partition names, expect to query from all partitions
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

	// query
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
	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
		} else {
			mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
		}
	}

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
		// common.CheckQueryResult(t, queryResultMultiPartition, []entity.Column{entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0})})
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

		// query with empty output fields []string{}-> output "int64"
		queryEmptyOutputs, _ := mc.QueryByPks(
			ctx, collName, []string{common.DefaultPartition},
			ids.Slice(0, 10),
			[]string{},
		)
		common.CheckOutputFields(t, queryEmptyOutputs, []string{common.DefaultIntFieldName})

		// query with empty output fields []string{""}-> output "int64" and dynamic field
		queryEmptyOutputs, err := mc.QueryByPks(
			ctx, collName, []string{common.DefaultPartition},
			ids.Slice(0, 10),
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
			ids.Slice(0, 10),
			[]string{common.DefaultFloatFieldName},
		)
		common.CheckOutputFields(t, queryFloatOutputs, []string{common.DefaultIntFieldName, common.DefaultFloatFieldName})
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

	// query
	_, errQuery := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		ids.Slice(0, 10),
		[]string{common.DefaultIntFieldName, "varchar"},
	)
	common.CheckErr(t, errQuery, false, "field varchar not exist")
}

// test query empty output fields: []string{} -> default pk
// test query empty output fields: []string{""} -> error
// test query with not existed field ["aa"]: error or as dynamic field
// test query with part not existed field ["aa", "$meat"]: error or as dynamic field
// test query with repeated field: ["*", "$meat"], ["floatVec", floatVec"] unique field
func TestQueryEmptyOutputFields2(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, enableDynamic := range []bool{true, false} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: enableDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: enableDynamic,
		}
		_, _ = insertData(ctx, t, mc, dp)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// query with empty output fields []string{}-> output "int64"
		expr := fmt.Sprintf("%s < 10", common.DefaultIntFieldName)
		queryNilOutputs, err := mc.Query(ctx, collName, []string{}, expr, []string{}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
		common.CheckErr(t, err, true)
		common.CheckOutputFields(t, queryNilOutputs, []string{common.DefaultIntFieldName})

		// query with not existed field -> output field as dynamic or error
		fakeName := "aaa"
		res2, err2 := mc.Query(ctx, collName, []string{}, expr, []string{fakeName}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
		if enableDynamic {
			common.CheckErr(t, err2, true)
			common.CheckOutputFields(t, res2, []string{common.DefaultIntFieldName, fakeName})
		} else {
			common.CheckErr(t, err2, false, fmt.Sprintf("%s not exist", fakeName))
		}

		// query with part not existed field ["aa", "$meat"]: error or as dynamic field
		res3, err3 := mc.Query(ctx, collName, []string{}, expr, []string{fakeName, common.DefaultDynamicFieldName}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
		if enableDynamic {
			common.CheckErr(t, err3, true)
			common.CheckOutputFields(t, res3, []string{common.DefaultIntFieldName, fakeName, common.DefaultDynamicFieldName})
		} else {
			common.CheckErr(t, err3, false, "not exist")
		}

		// query with repeated field: ["*", "$meat"], ["floatVec", floatVec"] unique field
		res4, err4 := mc.Query(ctx, collName, []string{}, expr, []string{"*", common.DefaultDynamicFieldName}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
		if enableDynamic {
			common.CheckErr(t, err4, true)
			common.CheckOutputFields(t, res4, []string{common.DefaultIntFieldName, common.DefaultFloatVecFieldName, common.DefaultFloatFieldName, common.DefaultDynamicFieldName})
		} else {
			common.CheckErr(t, err4, false, "$meta not exist")
		}

		res5, err5 := mc.Query(ctx, collName, []string{}, expr, []string{common.DefaultFloatVecFieldName, common.DefaultFloatVecFieldName, common.DefaultIntFieldName}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))

		common.CheckErr(t, err5, true)
		common.CheckOutputFields(t, res5, []string{common.DefaultIntFieldName, common.DefaultFloatVecFieldName})
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

	// query
	pos := 10
	queryResult, _ := mc.QueryByPks(
		ctx, collName,
		[]string{common.DefaultPartition},
		intColumn.Slice(0, pos),
		[]string{common.DefaultIntFieldName, common.DefaultFloatFieldName, common.DefaultFloatVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		intColumn.Slice(0, pos),
		floatColumn.Slice(0, pos),
		vecColumn.Slice(0, pos),
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

	// query
	pos := 10
	queryResult, _ := mc.QueryByPks(
		ctx,
		collName,
		[]string{common.DefaultPartition},
		varcharColumn.Slice(0, pos),
		[]string{common.DefaultBinaryVecFieldName},
	)
	common.CheckQueryResult(t, queryResult, []entity.Column{
		varcharColumn.Slice(0, pos),
		vecColumn.Slice(0, pos),
	})
	common.CheckOutputFields(t, queryResult, []string{common.DefaultBinaryVecFieldName, common.DefaultVarcharFieldName})
}

// test query output all fields
func TestOutputAllFieldsRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	var capacity int64 = common.TestCapacity
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity,
	}
	collName := createCollection(ctx, t, mc, cp)

	// prepare and insert data
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: true,
	}
	_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))

	// flush and check row count
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	for _, fieldName := range []string{"floatVec", "fp16Vec", "bf16Vec"} {
		_ = mc.CreateIndex(ctx, collName, fieldName, idx, false)
	}
	binIdx, _ := entity.NewIndexBinFlat(entity.JACCARD, 16)
	_ = mc.CreateIndex(ctx, collName, "binaryVec", binIdx, false)

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

// test query output all fields and verify data
func TestOutputAllFieldsColumn(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	var capacity int64 = common.TestCapacity
	for _, isDynamic := range [2]bool{true, false} {
		cp := CollectionParams{
			CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: isDynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity,
		}
		collName := createCollection(ctx, t, mc, cp)

		// prepare and insert data
		data := common.GenAllFieldsData(0, common.DefaultNb, common.DefaultDim, common.WithArrayCapacity(10))
		_data := data
		if isDynamic {
			_data = append(_data, common.GenDynamicFieldData(0, common.DefaultNb)...)
		}
		ids, err := mc.Insert(ctx, collName, "", _data...)
		common.CheckErr(t, err, true)
		require.Equal(t, common.DefaultNb, ids.Len())

		// flush and check row count
		errFlush := mc.Flush(ctx, collName, false)
		common.CheckErr(t, errFlush, true)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		for _, fieldName := range []string{"floatVec", "fp16Vec", "bf16Vec"} {
			_ = mc.CreateIndex(ctx, collName, fieldName, idx, false)
		}
		binIdx, _ := entity.NewIndexBinFlat(entity.JACCARD, 16)
		_ = mc.CreateIndex(ctx, collName, "binaryVec", binIdx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// query output all fields -> output all fields, includes vector and $meta field
		pos := 10
		allFieldsName := append(common.AllArrayFieldsName, "int64", "bool", "int8", "int16", "int32", "float",
			"double", "varchar", "json", "floatVec", "fp16Vec", "bf16Vec", "binaryVec")
		if isDynamic {
			allFieldsName = append(allFieldsName, common.DefaultDynamicFieldName)
		}
		queryResultAll, errQuery := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s < %d", common.DefaultIntFieldName, pos), []string{"*"})
		common.CheckErr(t, errQuery, true)
		common.CheckOutputFields(t, queryResultAll, allFieldsName)

		expColumns := make([]entity.Column, 0, len(data)+1)
		for _, column := range data {
			expColumns = append(expColumns, column.Slice(0, pos))
		}
		if isDynamic {
			expColumns = append(expColumns, common.MergeColumnsToDynamic(pos, common.GenDynamicFieldData(0, pos), common.DefaultDynamicFieldName))
		}
		common.CheckQueryResult(t, queryResultAll, expColumns)
	}
}

// Test query json collection, filter json field, output json field
func TestQueryJsonDynamicField(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	for _, dynamicField := range []bool{true, false} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: dynamicField,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: dynamicField,
		}
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
			dynamicColumn := common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2), common.DefaultDynamicFieldName)
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
			common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2), common.DefaultDynamicFieldName),
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
		cp := CollectionParams{
			CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: capacity,
		}
		collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

		// prepare and insert data
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows,
		}
		_, _ = insertData(ctx, t, mc, dp, common.WithArrayCapacity(capacity))

		// flush and check row count
		errFlush := mc.Flush(ctx, collName, false)
		common.CheckErr(t, errFlush, true)

		// index
		indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
		for _, fieldName := range common.AllVectorsFieldsName {
			if fieldName == common.DefaultBinaryVecFieldName {
				err := mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
				common.CheckErr(t, err, true)
			} else {
				err := mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
				common.CheckErr(t, err, true)
			}
		}

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
			log.Println(countRes.GetColumn(common.QueryCountFieldName).FieldData())
			common.CheckErr(t, err, true)
			require.Equal(t, _exprCount.count, countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])
		}

		// build BITMAP scalar index and query
		// release collection
		errRelease := mc.ReleaseCollection(ctx, collName)
		common.CheckErr(t, errRelease, true)

		BitmapNotSupportFiledNames := []interface{}{common.DefaultFloatArrayField, common.DefaultDoubleArrayField}
		scalarIdx := entity.NewScalarIndexWithType(entity.Bitmap)
		collection, _ := mc.DescribeCollection(ctx, collName)
		common.PrintAllFieldNames(collName, collection.Schema)
		for _, field := range collection.Schema.Fields {
			if field.DataType == entity.FieldTypeArray && !common.CheckContainsValue(BitmapNotSupportFiledNames, field.Name) {
				// create BITMAP scalar index
				err := mc.CreateIndex(ctx, collName, field.Name, scalarIdx, false, client.WithIndexName(field.Name+"scalar_index"))
				common.CheckErr(t, err, true)
			}
		}

		// load collection
		errLoad = mc.LoadCollection(ctx, collName, true)
		common.CheckErr(t, errLoad, true)

		for _, _exprCount := range exprCounts {
			log.Println(_exprCount.expr)
			countRes, err := mc.Query(ctx, collName,
				[]string{}, _exprCount.expr, []string{common.QueryCountFieldName})
			log.Println(countRes.GetColumn(common.QueryCountFieldName).FieldData())
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

	// verify only dynamic part key: common.DefaultDynamicNumberField
	dynamicNumColumn := queryRes.GetColumn(common.DefaultDynamicNumberField)
	numberValues := make([]int32, 0, 10)
	for i := 0; i < 10; i++ {
		numberValues = append(numberValues, int32(i))
	}
	_expColumn := common.MergeColumnsToDynamic(10, common.GenDynamicFieldData(0, 10)[:1], common.DefaultDynamicFieldName)
	expColumn := entity.NewColumnDynamic(_expColumn, common.DefaultDynamicNumberField)
	common.EqualColumn(t, dynamicNumColumn, expColumn)
}

// test query and output both json and dynamic field
func TestQueryJsonDynamicFieldRows(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: true,
	}
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
	m0 := common.JSONStruct{String: strconv.Itoa(0), Bool: true}
	j0, _ := json.Marshal(&m0)
	m1 := common.JSONStruct{Number: int32(1), String: strconv.Itoa(1), Bool: false, List: []int64{int64(1), int64(2)}}
	j1, _ := json.Marshal(&m1)
	jsonValues := [][]byte{j0, j1}
	jsonColumn := entity.NewColumnJSONBytes(common.DefaultJSONFieldName, jsonValues)
	dynamicColumn := common.MergeColumnsToDynamic(10, common.GenDynamicFieldData(0, 10), common.DefaultDynamicFieldName)
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
	common.CheckQueryResult(t, queryResult, []entity.Column{pkColumn, jsonColumn, dynamicColumn.Slice(0, 2)})

	// query with different expr and count
	expr := fmt.Sprintf("%s['number'] < 10 && %s < 10", common.DefaultJSONFieldName, common.DefaultDynamicNumberField)
	queryRes, _ := mc.Query(ctx, collName,
		[]string{common.DefaultPartition}, expr, []string{common.DefaultDynamicNumberField})

	// verify output fields and count, dynamicNumber value
	common.CheckOutputFields(t, queryRes, []string{common.DefaultIntFieldName, common.DefaultDynamicNumberField})
	pkColumn2 := common.GenColumnData(0, 10, entity.FieldTypeInt64, common.DefaultIntFieldName)
	common.CheckQueryResult(t, queryRes, []entity.Column{pkColumn2, dynamicColumn})
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
		common.CheckErr(t, err, _invalidExprs.ErrNil, _invalidExprs.ErrMsg, "invalid parameter")
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

		// query with empty output fields []string{}-> output "int64"
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

	// inert 1000 entities -> count*
	insertNb := 1000
	dpInsert := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: common.DefaultNb, nb: insertNb, dim: common.DefaultDim, EnableDynamicField: true,
	}
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

func TestQuerySparseVector(t *testing.T) {
	t.Parallel()
	idxInverted := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_INVERTED_INDEX", map[string]string{"drop_ratio_build": "0.2", "metric_type": "IP"})
	idxWand := entity.NewGenericIndex(common.DefaultSparseVecFieldName, "SPARSE_WAND", map[string]string{"drop_ratio_build": "0.3", "metric_type": "IP"})
	for _, idx := range []entity.Index{idxInverted, idxWand} {
		ctx := createContext(t, time.Second*common.DefaultTimeout*2)
		// connect
		mc := createMilvusClient(ctx, t)

		// create -> insert [0, 3000) -> flush -> index -> load
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: false,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen,
		}
		collName := createCollection(ctx, t, mc, cp)

		// index
		idxHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idxHnsw, false)
		mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, idx, false)

		// insert
		intColumn, _, floatColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
		varColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeVarChar, common.DefaultVarcharFieldName)
		sparseColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeSparseVector, common.DefaultSparseVecFieldName, common.WithSparseVectorLen(20))
		mc.Insert(ctx, collName, "", intColumn, varColumn, floatColumn, sparseColumn)
		mc.Flush(ctx, collName, false)
		mc.LoadCollection(ctx, collName, false)

		// count(*)
		countRes, _ := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s >=0", common.DefaultIntFieldName), []string{common.QueryCountFieldName})
		require.Equal(t, int64(common.DefaultNb), countRes.GetColumn(common.QueryCountFieldName).(*entity.ColumnInt64).Data()[0])

		// query
		queryResult, err := mc.Query(ctx, collName, []string{}, fmt.Sprintf("%s in [0, 1]", common.DefaultIntFieldName), []string{"*"})
		common.CheckErr(t, err, true)
		common.CheckOutputFields(t, queryResult, []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName, common.DefaultFloatVecFieldName, common.DefaultSparseVecFieldName})
		t.Log("https://github.com/milvus-io/milvus-sdk-go/issues/769")
		// common.CheckQueryResult(t, queryResult, []entity.Column{intColumn.Slice(0, 2), varColumn.Slice(0, 2), floatColumn.Slice(0, 2), sparseColumn.Slice(0, 2)})
	}
}

// test query iterator default
func TestQueryIteratorDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	dp2 := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
		start: common.DefaultNb, nb: common.DefaultNb * 2, dim: common.DefaultDim, EnableDynamicField: true, WithRows: true,
	}
	_, _ = insertData(ctx, t, mc, dp2)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query iterator with default batch
	itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName))
	common.CheckErr(t, err, true)
	common.CheckQueryIteratorResult(ctx, t, itr, common.DefaultNb*3, common.WithExpBatchSize(common.GenBatchSizes(common.DefaultNb*3, common.DefaultBatchSize)))
}

// test query iterator default
func TestQueryIteratorHitEmpty(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query iterator with default batch
	itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName))
	common.CheckErr(t, err, true)
	rs, err := itr.Next(ctx)
	require.Empty(t, rs)
	require.Error(t, err, io.EOF)
	common.CheckQueryIteratorResult(ctx, t, itr, 0, common.WithExpBatchSize(common.GenBatchSizes(0, common.DefaultBatchSize)))
}

func TestQueryIteratorBatchSize(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	nb := 201
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
		start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	type batchStruct struct {
		batch        int
		expBatchSize []int
	}
	batchStructs := []batchStruct{
		{batch: nb / 2, expBatchSize: common.GenBatchSizes(nb, nb/2)},
		{batch: nb, expBatchSize: common.GenBatchSizes(nb, nb)},
		{batch: nb + 1, expBatchSize: common.GenBatchSizes(nb, nb+1)},
	}

	for _, _batchStruct := range batchStructs {
		// query iterator with default batch
		itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithBatchSize(_batchStruct.batch))
		common.CheckErr(t, err, true)
		common.CheckQueryIteratorResult(ctx, t, itr, nb, common.WithExpBatchSize(_batchStruct.expBatchSize))
	}
}

func TestQueryIteratorOutputAllFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, dynamic := range [2]bool{false, true} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: dynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

		// insert
		nb := 2501
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
			start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: dynamic, WithRows: false,
		}
		insertData(ctx, t, mc, dp)

		indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
		for _, fieldName := range common.AllVectorsFieldsName {
			if fieldName == common.DefaultBinaryVecFieldName {
				mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
			} else {
				mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
			}
		}

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// output * fields
		nbFilter := 1001
		batch := 500
		expr := fmt.Sprintf("%s < %d", common.DefaultIntFieldName, nbFilter)

		itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithBatchSize(batch).WithOutputFields("*").WithExpr(expr))
		common.CheckErr(t, err, true)
		allFields := common.GetAllFieldsName(dynamic, false)
		common.CheckQueryIteratorResult(ctx, t, itr, nbFilter, common.WithExpBatchSize(common.GenBatchSizes(nbFilter, batch)), common.WithExpOutputFields(allFields))
	}
}

func TestQueryIteratorOutputSparseFieldsRows(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, withRows := range [2]bool{true, false} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: Int64VarcharSparseVec, AutoID: false, EnableDynamicField: true,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxLength: common.TestMaxLen,
		}
		collName := createCollection(ctx, t, mc, cp)

		// insert
		nb := 2501
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64VarcharSparseVec,
			start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: withRows, maxLenSparse: 1000,
		}
		_, _ = insertData(ctx, t, mc, dp)

		indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		indexSparse, _ := entity.NewIndexSparseInverted(entity.IP, 0.1)
		mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, indexHnsw, false)
		mc.CreateIndex(ctx, collName, common.DefaultSparseVecFieldName, indexSparse, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// output * fields
		itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithBatchSize(400).WithOutputFields("*"))
		common.CheckErr(t, err, true)
		fields := []string{common.DefaultIntFieldName, common.DefaultVarcharFieldName, common.DefaultFloatVecFieldName, common.DefaultSparseVecFieldName, common.DefaultDynamicFieldName}
		common.CheckQueryIteratorResult(ctx, t, itr, nb, common.WithExpBatchSize(common.GenBatchSizes(nb, 400)), common.WithExpOutputFields(fields))
	}
}

// test query iterator with non-existed collection/partition name, invalid batch size
func TestQueryIteratorInvalid(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	nb := 201
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
		start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: false, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query iterator with not existed collection name
	_, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption("aaa"))
	common.CheckErr(t, err, false, "can't find collection")

	// query iterator with not existed partition name
	_, errPar := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithPartitions("aaa"))
	common.CheckErr(t, errPar, false, "partition name aaa not found")

	// query iterator with not existed partition name
	_, errPar = mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithPartitions("aaa", common.DefaultPartition))
	common.CheckErr(t, errPar, false, "partition name aaa not found")

	_, errOutput := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithOutputFields(common.QueryCountFieldName))
	common.CheckErr(t, errOutput, false, "count entities with pagination is not allowed")

	// query iterator with invalid batch size
	for _, batch := range []int{-1, 0} {
		// query iterator with default batch
		_, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithBatchSize(batch))
		common.CheckErr(t, err, false, "batch size cannot less than 1")
	}
}

// query iterator with invalid expr
func TestQueryIteratorInvalidExpr(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVecJSON, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVecJSON,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	for _, _invalidExprs := range common.InvalidExpressions {
		t.Log(_invalidExprs)
		_, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithExpr(_invalidExprs.Expr))
		common.CheckErr(t, err, _invalidExprs.ErrNil, "invalid parameter", _invalidExprs.ErrMsg)
	}
}

// test query iterator with non-existed field when dynamic or not
func TestQueryIteratorOutputFieldDynamic(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	for _, dynamic := range [2]bool{true, false} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: dynamic,
			ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
		}
		collName := createCollection(ctx, t, mc, cp)
		// insert
		nb := 201
		dp := DataParams{
			CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
			start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: dynamic, WithRows: false,
		}
		_, _ = insertData(ctx, t, mc, dp)

		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

		// Load collection
		errLoad := mc.LoadCollection(ctx, collName, false)
		common.CheckErr(t, errLoad, true)

		// query iterator with not existed output fields: if dynamic, non-existent field are equivalent to dynamic field
		itr, errOutput := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithOutputFields("aaa"))
		if dynamic {
			common.CheckErr(t, errOutput, true)
			expFields := []string{common.DefaultIntFieldName, common.DefaultDynamicFieldName}
			common.CheckQueryIteratorResult(ctx, t, itr, nb, common.WithExpBatchSize(common.GenBatchSizes(nb, common.DefaultBatchSize)), common.WithExpOutputFields(expFields))
		} else {
			common.CheckErr(t, errOutput, false, "field aaa not exist")
		}
	}
}

func TestQueryIteratorExpr(t *testing.T) {
	// t.Log("https://github.com/milvus-io/milvus-sdk-go/issues/756")
	type exprCount struct {
		expr  string
		count int
	}
	capacity := common.TestCapacity
	exprLimits := []exprCount{
		{expr: fmt.Sprintf("%s in [0, 1, 2]", common.DefaultIntFieldName), count: 3},
		{expr: fmt.Sprintf("%s >= 1000 || %s > 2000", common.DefaultIntFieldName, common.DefaultIntFieldName), count: 2000},
		{expr: fmt.Sprintf("%s >= 1000 and %s < 2000", common.DefaultIntFieldName, common.DefaultIntFieldName), count: 1000},

		// json and dynamic field filter expr: == < in bool/ list/ int
		{expr: fmt.Sprintf("%s['number'] == 0", common.DefaultJSONFieldName), count: 1500 / 2},
		{expr: fmt.Sprintf("%s['number'] < 100 and %s['number'] != 0", common.DefaultJSONFieldName, common.DefaultJSONFieldName), count: 50},
		{expr: fmt.Sprintf("%s < 100", common.DefaultDynamicNumberField), count: 100},
		{expr: "dynamicNumber % 2 == 0", count: 1500},
		{expr: fmt.Sprintf("%s == false", common.DefaultDynamicBoolField), count: 2000},
		{expr: fmt.Sprintf("%s in ['1', '2'] ", common.DefaultDynamicStringField), count: 2},
		{expr: fmt.Sprintf("%s['string'] in ['1', '2', '5'] ", common.DefaultJSONFieldName), count: 3},
		{expr: fmt.Sprintf("%s['list'] == [1, 2] ", common.DefaultJSONFieldName), count: 1},
		{expr: fmt.Sprintf("%s['list'][0] < 10 ", common.DefaultJSONFieldName), count: 5},
		{expr: fmt.Sprintf("%s[\"dynamicList\"] != [2, 3]", common.DefaultDynamicFieldName), count: 0},

		// json contains
		{expr: fmt.Sprintf("json_contains (%s['list'], 2)", common.DefaultJSONFieldName), count: 1},
		{expr: fmt.Sprintf("json_contains (%s['number'], 0)", common.DefaultJSONFieldName), count: 0},
		{expr: fmt.Sprintf("JSON_CONTAINS_ANY (%s['list'], [1, 3])", common.DefaultJSONFieldName), count: 2},
		// string like
		{expr: "dynamicString like '1%' ", count: 1111},

		// key exist
		{expr: fmt.Sprintf("exists %s['list']", common.DefaultJSONFieldName), count: common.DefaultNb / 2},
		{expr: fmt.Sprintf("exists a "), count: 0},
		{expr: fmt.Sprintf("exists %s ", common.DefaultDynamicStringField), count: common.DefaultNb},

		// data type not match and no error
		{expr: fmt.Sprintf("%s['number'] == '0' ", common.DefaultJSONFieldName), count: 0},

		// json field
		{expr: fmt.Sprintf("%s >= 1500", common.DefaultJSONFieldName), count: 1500 / 2},                                  // json >= 1500
		{expr: fmt.Sprintf("%s > 1499.5", common.DefaultJSONFieldName), count: 1500 / 2},                                 // json >= 1500.0
		{expr: fmt.Sprintf("%s like '21%%'", common.DefaultJSONFieldName), count: 100 / 4},                               // json like '21%'
		{expr: fmt.Sprintf("%s == [1503, 1504]", common.DefaultJSONFieldName), count: 1},                                 // json == [1,2]
		{expr: fmt.Sprintf("%s[0] > 1", common.DefaultJSONFieldName), count: 1500 / 4},                                   // json[0] > 1
		{expr: fmt.Sprintf("%s[0][0] > 1", common.DefaultJSONFieldName), count: 0},                                       // json == [1,2]
		{expr: fmt.Sprintf("%s[0] == false", common.DefaultBoolArrayField), count: common.DefaultNb / 2},                 //  array[0] ==
		{expr: fmt.Sprintf("%s[0] > 0", common.DefaultInt64ArrayField), count: common.DefaultNb - 1},                     //  array[0] >
		{expr: fmt.Sprintf("%s[0] > 0", common.DefaultInt8ArrayField), count: 1524},                                      //  array[0] > int8 range: [-128, 127]
		{expr: fmt.Sprintf("array_contains (%s, %d)", common.DefaultInt16ArrayField, capacity), count: capacity},         // array_contains(array, 1)
		{expr: fmt.Sprintf("json_contains (%s, 1)", common.DefaultInt32ArrayField), count: 2},                            // json_contains(array, 1)
		{expr: fmt.Sprintf("array_contains (%s, 1000000)", common.DefaultInt32ArrayField), count: 0},                     // array_contains(array, 1)
		{expr: fmt.Sprintf("json_contains_all (%s, [90, 91])", common.DefaultInt64ArrayField), count: 91},                // json_contains_all(array, [x])
		{expr: fmt.Sprintf("json_contains_any (%s, [0, 100, 10])", common.DefaultFloatArrayField), count: 101},           // json_contains_any (array, [x])
		{expr: fmt.Sprintf("%s == [0, 1]", common.DefaultDoubleArrayField), count: 0},                                    //  array ==
		{expr: fmt.Sprintf("array_length(%s) == %d", common.DefaultDoubleArrayField, capacity), count: common.DefaultNb}, //  array_length
	}

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// create collection
	cp := CollectionParams{
		CollectionFieldsType: AllFields, AutoID: false, EnableDynamicField: true,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim, MaxCapacity: common.TestCapacity,
	}
	collName := createCollection(ctx, t, mc, cp, client.WithConsistencyLevel(entity.ClStrong))

	// insert
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: AllFields,
		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false,
	}
	_, err := insertData(ctx, t, mc, dp, common.WithArrayCapacity(common.TestCapacity))
	common.CheckErr(t, err, true)
	mc.Flush(ctx, collName, false)

	indexHnsw, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	indexBinary, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 64)
	for _, fieldName := range common.AllVectorsFieldsName {
		if fieldName == common.DefaultBinaryVecFieldName {
			mc.CreateIndex(ctx, collName, fieldName, indexBinary, false)
		} else {
			mc.CreateIndex(ctx, collName, fieldName, indexHnsw, false)
		}
	}

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)
	batch := 500

	for _, exprLimit := range exprLimits {
		log.Printf("case expr is: %s, limit=%d", exprLimit.expr, exprLimit.count)
		itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithBatchSize(batch).WithExpr(exprLimit.expr))
		common.CheckErr(t, err, true)
		common.CheckQueryIteratorResult(ctx, t, itr, exprLimit.count, common.WithExpBatchSize(common.GenBatchSizes(exprLimit.count, batch)))
	}
}

// test query iterator with partition
func TestQueryIteratorPartitions(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	// create collection
	cp := CollectionParams{
		CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: false,
		ShardsNum: common.DefaultShards, Dim: common.DefaultDim,
	}
	collName := createCollection(ctx, t, mc, cp)
	pName := "p1"
	err := mc.CreatePartition(ctx, collName, pName)
	common.CheckErr(t, err, true)

	// insert [0, nb) into partition: _default
	nb := 1500
	dp := DataParams{
		CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
		start: 0, nb: nb, dim: common.DefaultDim, EnableDynamicField: false, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp)
	// insert [nb, nb*2) into partition: p1
	dp1 := DataParams{
		CollectionName: collName, PartitionName: pName, CollectionFieldsType: Int64FloatVec,
		start: nb, nb: nb, dim: common.DefaultDim, EnableDynamicField: false, WithRows: false,
	}
	_, _ = insertData(ctx, t, mc, dp1)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query iterator with partition
	expr := fmt.Sprintf("%s < %d", common.DefaultIntFieldName, nb)
	mParLimit := map[string]int{
		common.DefaultPartition: nb,
		pName:                   0,
	}
	for par, limit := range mParLimit {
		itr, err := mc.QueryIterator(ctx, client.NewQueryIteratorOption(collName).WithExpr(expr).WithPartitions(par))
		common.CheckErr(t, err, true)
		common.CheckQueryIteratorResult(ctx, t, itr, limit, common.WithExpBatchSize(common.GenBatchSizes(limit, common.DefaultBatchSize)))
	}
}

// TODO offset and limit
// TODO consistency level
// TODO ignore growing
