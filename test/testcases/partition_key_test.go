//go:build L0

package testcases

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test enable partition key with int64 field
func TestPartitionKeyDefaultInt64(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// fields
	partitionKeyFieldName := "partitionKeyField"
	partitionKeyField := common.GenField(partitionKeyFieldName, entity.FieldTypeInt64,
		common.WithIsPartitionKey(true), common.WithMaxLength(common.MaxLength))

	// schema
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false))
	schema.WithField(partitionKeyField)

	// create collection and check partition key
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10))
	common.CheckErr(t, err, true)

	// insert data
	partitionKeyColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeInt64, partitionKeyFieldName)
	pkColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, schema.CollectionName, "", pkColumn, floatColumn, vecColumn, partitionKeyColumn)
	common.CheckErr(t, errInsert, true)
	require.Equalf(t, common.DefaultNb, ids.Len(), fmt.Sprintf("Expected insert result equal to %d, actual %d", common.DefaultNb, ids.Len()))

	// flush and create index and load collection
	mc.Flush(ctx, schema.CollectionName, true)
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, schema.CollectionName, common.DefaultFloatVecFieldName, idx, false)
	mc.LoadCollection(ctx, schema.CollectionName, false)

	// query filter partition key field and other field
	queryIds := []entity.Column{
		common.GenColumnData(0, 10, entity.FieldTypeInt64, partitionKeyFieldName),
		common.GenColumnData(0, 10, entity.FieldTypeInt64, common.DefaultIntFieldName),
	}
	for _, queryID := range queryIds {
		queryResult, errQuery := mc.QueryByPks(ctx, schema.CollectionName, []string{},
			queryID,
			[]string{common.DefaultIntFieldName},
		)
		common.CheckErr(t, errQuery, true)
		common.CheckQueryResult(t, queryResult, []entity.Column{common.GenColumnData(0, 10, entity.FieldTypeInt64, common.DefaultIntFieldName)})
	}

	// search vector with expr: in, ==, >, non-partition-key field
	sp, _ := entity.NewIndexHNSWSearchParam(64)
	exprs := []string{
		fmt.Sprintf("%s < 1000", partitionKeyFieldName),
		fmt.Sprintf("%s == 1000", partitionKeyFieldName),
		fmt.Sprintf("%s in [99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 300, 789, 525, 22]", partitionKeyFieldName),
		fmt.Sprintf("%s < 1000.0 && %s > 5", common.DefaultFloatFieldName, partitionKeyFieldName),
	}
	for _, expr := range exprs {
		// expr filter
		searchResult, errSearch := mc.Search(
			ctx, schema.CollectionName,
			[]string{},
			expr,
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearch, true)
		if strings.Contains(expr, "==") {
			common.CheckSearchResult(t, searchResult, common.DefaultNq, 1)
		} else {
			common.CheckSearchResult(t, searchResult, common.DefaultNq, common.DefaultTopK)
		}
		for _, res := range searchResult {
			for _, id := range res.IDs.(*entity.ColumnInt64).Data() {
				require.LessOrEqualf(t, id, int64(1000), "The id search returned is expected to <= 1000")
			}
		}
	}
}

// test enable partition key with varchar field
func TestPartitionKeyDefaultVarchar(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// fields
	partitionKeyFieldName := "partitionKeyField"
	partitionKeyField := common.GenField(partitionKeyFieldName, entity.FieldTypeVarChar,
		common.WithIsPartitionKey(true), common.WithMaxLength(common.MaxLength))

	// schema
	schema := common.GenSchema(common.GenRandomString(6), false, common.GenDefaultFields(false))
	schema.WithField(partitionKeyField)

	// create collection and check partition key
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10))
	common.CheckErr(t, err, true)

	// insert data
	partitionKeyColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeVarChar, partitionKeyFieldName)
	pkColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, schema.CollectionName, "", pkColumn, floatColumn, vecColumn, partitionKeyColumn)
	common.CheckErr(t, errInsert, true)
	require.Equalf(t, common.DefaultNb, ids.Len(), fmt.Sprintf("Expected insert result equal to %d, actual %d", common.DefaultNb, ids.Len()))

	// flush and create index and load collection
	mc.Flush(ctx, schema.CollectionName, true)
	idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
	mc.CreateIndex(ctx, schema.CollectionName, common.DefaultFloatVecFieldName, idx, false)
	mc.LoadCollection(ctx, schema.CollectionName, false)

	// query filter partition key field and other field
	queryIds := []entity.Column{
		common.GenColumnData(0, 10, entity.FieldTypeVarChar, partitionKeyFieldName),
		common.GenColumnData(0, 10, entity.FieldTypeInt64, common.DefaultIntFieldName),
	}
	for _, queryID := range queryIds {
		queryResult, errQuery := mc.QueryByPks(ctx, schema.CollectionName, []string{},
			queryID,
			[]string{common.DefaultIntFieldName},
		)
		common.CheckErr(t, errQuery, true)
		common.CheckQueryResult(t, queryResult, []entity.Column{common.GenColumnData(0, 10, entity.FieldTypeInt64, common.DefaultIntFieldName)})
	}

	// search vector with expr: in, ==, >, non-partition-key field
	sp, _ := entity.NewIndexHNSWSearchParam(64)
	exprs := []string{
		fmt.Sprintf("%s < '9'", partitionKeyFieldName),
		fmt.Sprintf("%s == '1000'", partitionKeyFieldName),
		fmt.Sprintf("%s in ['99', '199', '299', '399', '499', '599', '699', '799', '899', '999', '300', '789', '525', '22']", partitionKeyFieldName),
		fmt.Sprintf("%s < 1000.0 && %s > '5'", common.DefaultFloatFieldName, partitionKeyFieldName),
	}
	for _, expr := range exprs {
		// expr filter
		searchResult, errSearch := mc.Search(
			ctx, schema.CollectionName,
			[]string{},
			expr,
			[]string{common.DefaultIntFieldName},
			common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
			common.DefaultFloatVecFieldName,
			entity.L2,
			common.DefaultTopK,
			sp,
		)
		common.CheckErr(t, errSearch, true)
		if strings.Contains(expr, "==") {
			common.CheckSearchResult(t, searchResult, common.DefaultNq, 1)
		} else {
			common.CheckSearchResult(t, searchResult, common.DefaultNq, common.DefaultTopK)
		}
	}
}

func TestPartitionKeyInvalidNumPartition(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// prepare field and schema
	partitionKeyFieldName := "partitionKeyField"
	partitionKeyField := common.GenField(partitionKeyFieldName, entity.FieldTypeInt64, common.WithIsPartitionKey(true))

	// schema
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, common.GenDefaultFields(false))
	schema.WithField(partitionKeyField)

	invalidNumPartitionStruct := []struct {
		numPartitions int64
		errMsg        string
	}{
		{common.MaxPartitionNum + 1, "exceeds max configuration (4096)"},
		{-1, "the specified partitions should be greater than 0 if partition key is used"},
	}
	for _, npStruct := range invalidNumPartitionStruct {

		// create collection and check partition key
		err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(npStruct.numPartitions))
		common.CheckErr(t, err, false, npStruct.errMsg)
	}

	// PartitionNum is 0, actually default 64 partitions
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(0))
	common.CheckErr(t, err, true)
	partitions, _ := mc.ShowPartitions(ctx, collName)
	require.Equal(t, len(partitions), common.DefaultPartitionNum)
}

func TestPartitionKeyNumPartition(t *testing.T) {
	// test set num partition range [1, 4096]
	// set num partition
	t.Parallel()

	numPartitionsValues := []int64{
		1,
		128,
		64,
		4096,
	}
	for _, numPartitionsValue := range numPartitionsValues {
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		mc := createMilvusClient(ctx, t)

		// prepare field and schema
		partitionKeyFieldName := "partitionKeyField"
		partitionKeyField := common.GenField(partitionKeyFieldName, entity.FieldTypeInt64, common.WithIsPartitionKey(true))

		// schema
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, false, common.GenDefaultFields(false))
		schema.WithField(partitionKeyField)

		// create collection and check partition key
		err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(numPartitionsValue))
		common.CheckErr(t, err, true)

		// insert and query search
		collections, _ := mc.ListCollections(ctx)
		common.CheckContainsCollection(t, collections, collName)
	}

}

// test partition key on invalid field
func TestPartitionKeyNotSupportFieldType(t *testing.T) {
	t.Parallel()
	// current only support int64 and varchar field
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	notSupportPartitionKeyFieldType := []entity.FieldType{
		entity.FieldTypeBool,
		entity.FieldTypeInt8,
		entity.FieldTypeInt16,
		entity.FieldTypeInt32,
		entity.FieldTypeFloat,
		entity.FieldTypeDouble,
		entity.FieldTypeJSON,
		entity.FieldTypeBinaryVector,
		entity.FieldTypeFloatVector,
	}

	for _, pkf := range notSupportPartitionKeyFieldType {
		// prepare field and schema
		partitionKeyFieldName := "partitionKeyField"
		partitionKeyField := common.GenField(partitionKeyFieldName, pkf, common.WithIsPartitionKey(true))

		// schema
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, false, common.GenDefaultFields(false))
		schema.WithField(partitionKeyField)

		// create collection and check partition key
		err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10))
		common.CheckErr(t, err, false, "the data type of partition key should be Int64 or VarChar")
	}
}

// test multi partition key fields -> error
func TestPartitionKeyMultiFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// multi partition key field
	partitionKeyField1 := common.GenField("intPartitionKeyField", entity.FieldTypeInt64, common.WithIsPartitionKey(true))
	partitionKeyField2 := common.GenField("varcharPartitionKeyField", entity.FieldTypeInt64, common.WithIsPartitionKey(true))

	// schema
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, common.GenDefaultFields(false))
	schema.WithField(partitionKeyField1).WithField(partitionKeyField2)

	// create collection and check partition key
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10))
	common.CheckErr(t, err, false, "there are more than one partition key")
}

func TestPartitionNumWhenDisablePartitionKey(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// schema
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, common.GenDefaultFields(false))

	// create collection and check partition key
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(10))
	common.CheckErr(t, err, false, "num_partitions should only be specified with partition key field enabled")
}

// test operate partition related after enable partition key -> error expect has partition
func TestPartitionKeyPartitionOperation(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// multi partition key field
	partitionKeyField := common.GenField("partitionKeyField", entity.FieldTypeInt64, common.WithIsPartitionKey(true))

	// schema
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, true, common.GenDefaultFields(true))
	schema.WithField(partitionKeyField)

	// create collection and check partition key
	partitionNum := 10
	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithPartitionNum(int64(partitionNum)))
	common.CheckErr(t, err, true)

	// list partitions -> success
	partitions, err := mc.ShowPartitions(ctx, collName)
	common.CheckErr(t, err, true)
	require.Lenf(t, partitions, partitionNum, fmt.Sprintf("Expected collection has %d partitions, actually %d.", partitionNum, len(partitions)))

	// has partition -> success
	has, err := mc.HasPartition(ctx, collName, partitions[0].Name)
	require.True(t, has)
	common.CheckErr(t, err, true)

	// create partition -> error
	err = mc.CreatePartition(ctx, collName, common.GenRandomString(4))
	common.CheckErr(t, err, false, "disable create partition if partition key mode is used")

	// drop partition -> error
	err = mc.DropPartition(ctx, collName, partitions[2].Name)
	common.CheckErr(t, err, false, "disable drop partition if partition key mode is used")

	// load partition -> error
	err = mc.LoadPartitions(ctx, collName, []string{partitions[0].Name}, true)
	common.CheckErr(t, err, false, "disable load partitions if partition key mode is used")

	// release partition -> error
	err = mc.ReleasePartitions(ctx, collName, []string{partitions[0].Name})
	common.CheckErr(t, err, false, "disable release partitions if partition key mode is used")

	// insert into partition -> error
	partitionKeyColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeInt64, partitionKeyField.Name)
	_, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	_, err = mc.Insert(ctx, schema.CollectionName, partitions[0].Name, floatColumn, vecColumn, partitionKeyColumn)
	common.CheckErr(t, err, false, "not support manually specifying the partition names if partition key mode is used")

	// InsertRows into partition -> error
	vector := make([]float32, 0, common.DefaultDim)
	for j := 0; j < int(common.DefaultDim); j++ {
		vector = append(vector, rand.Float32())
	}
	row := struct {
		Int64             int64     `json:"int64" milvus:"name:int64"`
		Float             float32   `json:"float" milvus:"name:float"`
		FloatVec          []float32 `json:"floatVec" milvus:"name:floatVec"`
		PartitionKeyField int64     `json:"partitionKeyField" milvus:"name:partitionKeyField"`
	}{int64(1), float32(1), vector, int64(2)}
	_, err = mc.InsertRows(ctx, collName, partitions[0].Name, []interface{}{row})
	common.CheckErr(t, err, false, "not support manually specifying the partition names if partition key mode is used")

	// delete from partition -> error
	err = mc.DeleteByPks(ctx, collName, partitions[2].Name, entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0, 1}))
	common.CheckErr(t, err, false, "not support manually specifying the partition names if partition key mode is used")

	// bulk insert -> error
	_, err = mc.BulkInsert(ctx, collName, partitions[0].Name, []string{""})
	common.CheckErr(t, err, false, "not allow to set partition name for collection with partition key: importing data failed")

	// query partitions -> error
	_, err = mc.QueryByPks(
		ctx, collName,
		[]string{partitions[0].Name},
		entity.NewColumnInt64(common.DefaultIntFieldName, []int64{0}), []string{})
	common.CheckErr(t, err, false, "not support manually specifying the partition names if partition key mode is used")

	// search
	sp, _ := entity.NewIndexHNSWSearchParam(74)
	_, err = mc.Search(
		ctx, collName, []string{partitions[0].Name}, "", []string{},
		common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector), common.DefaultFloatVecFieldName,
		entity.L2, 1, sp)
	common.CheckErr(t, err, false, "not support manually specifying the partition names if partition key mode is used")
}
