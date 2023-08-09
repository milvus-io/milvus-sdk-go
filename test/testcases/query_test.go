//go:build L0

package testcases

import (
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
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, false, true)

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
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, false, true)

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
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, false, true)

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
	collName := createDefaultCollection(ctx, t, mc, false, false)

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
	collName := createDefaultCollection(ctx, t, mc, false, false)
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
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, false, true)

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
// TODO "Issue: https://github.com/milvus-io/milvus-sdk-go/issues/366")
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
	for _, enableDynamic := range []bool{true} {
		// create, insert, index
		collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, enableDynamic, true)

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
			common.CheckOutputFields(t, queryEmptyOutputs, []string{"", common.DefaultIntFieldName})
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
	collName := createDefaultCollection(ctx, t, mc, false, false)
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

	queryResultAllScalar, errQuery := mc.QueryByPks(ctx, collName, []string{common.DefaultPartition},
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
	collName, ids := createCollectionWithDataIndex(ctx, t, mc, true, false, true)

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

	for _, dynamicField := range []bool{true} {
		// create collection
		cp := CollectionParams{
			CollectionFieldsType: Int64FloatVecJSON,
			AutoID:               false,
			EnableDynamicField:   dynamicField,
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
			EnableDynamicField:   dynamicField,
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

		// query and output json field
		pkColumn := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1})
		queryResult, err := mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			pkColumn,
			outputFields,
		)
		common.CheckErr(t, err, true)
		jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, 2)
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

		// query with json filter
		queryResult, err = mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			jsonColumn,
			[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName},
		)
		common.CheckErr(t, err, false, "only int64 and varchar column can be primary key for now")

		// query with dynamic field
		queryResult, err = mc.QueryByPks(
			ctx, collName,
			[]string{common.DefaultPartition},
			common.MergeColumnsToDynamic(2, common.GenDynamicFieldData(0, 2)),
			[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName},
		)
		common.CheckErr(t, err, false, "only int64 and varchar column can be primary key for now")
	}
}

// test query and output both json and dynamic field
func TestQueryJsonDynamicFieldRows(t *testing.T) {
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
		WithRows:             true,
	}
	_, _ = insertData(ctx, t, mc, dp)

	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

	// Load collection
	errLoad := mc.LoadCollection(ctx, collName, false)
	common.CheckErr(t, errLoad, true)

	// query and output json field
	pkColumn := entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1})
	queryResult, err := mc.QueryByPks(
		ctx, collName,
		[]string{common.DefaultPartition},
		pkColumn,
		[]string{common.DefaultIntFieldName, common.DefaultJSONFieldName, common.DefaultDynamicFieldName},
	)
	common.CheckErr(t, err, true)
	jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, 0, 2)
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
}

// TODO offset and limit
// TODO consistency level
// TODO ignore growing
