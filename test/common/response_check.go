package common

import (
	"context"
	"fmt"
	"io"
	"log"
	"reflect"
	"strings"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"
)

// CheckErr check err and errMsg
func CheckErr(t *testing.T, actualErr error, expErrNil bool, expErrorMsg ...string) {
	if expErrNil {
		require.NoError(t, actualErr)
	} else {
		require.Error(t, actualErr)
		switch len(expErrorMsg) {
		case 0:
			log.Fatal("expect error message should not be empty")
		case 1:
			require.ErrorContains(t, actualErr, expErrorMsg[0])
		default:
			contains := false
			for i := 0; i < len(expErrorMsg); i++ {
				if strings.Contains(actualErr.Error(), expErrorMsg[i]) {
					contains = true
				}
			}
			if !contains {
				t.Fatalf("CheckErr failed, actualErr doesn't contains any expErrorMsg, please check test cases!")
			}
		}
	}
}

// EqualFields equal two fields
func EqualFields(t *testing.T, fieldA *entity.Field, fieldB *entity.Field) {
	require.Equal(t, fieldA.Name, fieldB.Name, fmt.Sprintf("Expected field name: %s, actual: %s", fieldA.Name, fieldB.Name))
	require.Equal(t, fieldA.AutoID, fieldB.AutoID, fmt.Sprintf("Expected field AutoID: %t, actual: %t", fieldA.AutoID, fieldB.AutoID))
	require.Equal(t, fieldA.PrimaryKey, fieldB.PrimaryKey, fmt.Sprintf("Expected field PrimaryKey: %t, actual: %t", fieldA.PrimaryKey, fieldB.PrimaryKey))
	require.Equal(t, fieldA.Description, fieldB.Description, fmt.Sprintf("Expected field Description: %s, actual: %s", fieldA.Description, fieldB.Description))
	require.Equal(t, fieldA.DataType, fieldB.DataType, fmt.Sprintf("Expected field DataType: %v, actual: %v", fieldA.DataType, fieldB.DataType))
	require.Equal(t, fieldA.IsPartitionKey, fieldB.IsPartitionKey, fmt.Sprintf("Expected field IsPartitionKey: %t, actual: %t", fieldA.IsPartitionKey, fieldB.IsPartitionKey))
	require.Equal(t, fieldA.IsDynamic, fieldB.IsDynamic, fmt.Sprintf("Expected field IsDynamic: %t, actual: %t", fieldA.IsDynamic, fieldB.IsDynamic))

	// check vector field dim
	switch fieldA.DataType {
	case entity.FieldTypeFloatVector:
		require.Equal(t, fieldA.TypeParams[entity.TypeParamDim], fieldB.TypeParams[entity.TypeParamDim])
	case entity.FieldTypeBinaryVector:
		require.Equal(t, fieldA.TypeParams[entity.TypeParamDim], fieldB.TypeParams[entity.TypeParamDim])
	// check varchar field max_length
	case entity.FieldTypeVarChar:
		require.Equal(t, fieldA.TypeParams[entity.TypeParamMaxLength], fieldB.TypeParams[entity.TypeParamMaxLength])

	}
	require.Empty(t, fieldA.IndexParams)
	require.Empty(t, fieldB.IndexParams)
	// require.Equal(t, fieldA.IndexParams, fieldB.IndexParams)
}

// EqualSchema equal two schemas
func EqualSchema(t *testing.T, schemaA entity.Schema, schemaB entity.Schema) {
	require.Equal(t, schemaA.CollectionName, schemaB.CollectionName, fmt.Sprintf("Expected schame CollectionName: %s, actual: %s", schemaA.CollectionName, schemaB.CollectionName))
	require.Equal(t, schemaA.Description, schemaB.Description, fmt.Sprintf("Expected Description: %s, actual: %s", schemaA.Description, schemaB.Description))
	require.Equal(t, schemaA.AutoID, schemaB.AutoID, fmt.Sprintf("Expected schema AutoID: %t, actual: %t", schemaA.AutoID, schemaB.AutoID))
	require.Equal(t, len(schemaA.Fields), len(schemaB.Fields), fmt.Sprintf("Expected schame fields num: %d, actual: %d", len(schemaA.Fields), len(schemaB.Fields)))
	require.Equal(t, schemaA.EnableDynamicField, schemaB.EnableDynamicField, fmt.Sprintf("Expected schame EnableDynamicField: %t, actual: %t", schemaA.EnableDynamicField, schemaB.EnableDynamicField))
	for i := 0; i < len(schemaA.Fields); i++ {
		EqualFields(t, schemaA.Fields[i], schemaB.Fields[i])
	}
}

// CheckCollection check collection
func CheckCollection(t *testing.T, actualCollection *entity.Collection, expCollName string, expShardNum int32,
	expSchema *entity.Schema, expConsistencyLevel entity.ConsistencyLevel,
) {
	require.Equalf(t, expCollName, actualCollection.Name, fmt.Sprintf("Expected collection name: %s, actual: %v", expCollName, actualCollection.Name))
	require.Equalf(t, expShardNum, actualCollection.ShardNum, fmt.Sprintf("Expected ShardNum: %d, actual: %d", expShardNum, actualCollection.ShardNum))
	require.Equal(t, expConsistencyLevel, actualCollection.ConsistencyLevel, fmt.Sprintf("Expected ConsistencyLevel: %v, actual: %v", expConsistencyLevel, actualCollection.ConsistencyLevel))
	EqualSchema(t, *expSchema, *actualCollection.Schema)
}

// CheckContainsCollection check collections contains collName
func CheckContainsCollection(t *testing.T, collections []*entity.Collection, collName string) {
	allCollNames := make([]string, 0, len(collections))
	for _, collection := range collections {
		allCollNames = append(allCollNames, collection.Name)
	}
	require.Containsf(t, allCollNames, collName, fmt.Sprintf("The collection %s not in: %v", collName, allCollNames))
}

// CheckNotContainsCollection check collections not contains collName
func CheckNotContainsCollection(t *testing.T, collections []*entity.Collection, collName string) {
	allCollNames := make([]string, 0, len(collections))
	for _, collection := range collections {
		allCollNames = append(allCollNames, collection.Name)
	}
	require.NotContainsf(t, allCollNames, collName, fmt.Sprintf("The collection %s should not be in: %v", collName, allCollNames))
}

// CheckInsertResult check insert result, ids len (insert count), ids data (pks, but no auto ids)
func CheckInsertResult(t *testing.T, actualIDs entity.Column, expIDs entity.Column) {
	require.Equal(t, actualIDs.Len(), expIDs.Len())
	switch expIDs.Type() {
	// pk field support int64 and varchar type
	case entity.FieldTypeInt64:
		require.ElementsMatch(t, actualIDs.(*entity.ColumnInt64).Data(), expIDs.(*entity.ColumnInt64).Data())
	case entity.FieldTypeVarChar:
		require.ElementsMatch(t, actualIDs.(*entity.ColumnVarChar).Data(), expIDs.(*entity.ColumnVarChar).Data())
	default:
		log.Printf("The primary field only support type: [%v, %v]", entity.FieldTypeInt64, entity.FieldTypeVarChar)
	}
}

// CheckIndexResult check index result, index type, metric type, index params
func CheckIndexResult(t *testing.T, actualIndexes []entity.Index, expIndexes ...entity.Index) {
	mNameActualIndex := make(map[string]entity.Index)
	allActualIndexNames := make([]string, 0, len(actualIndexes))
	for _, actualIndex := range actualIndexes {
		mNameActualIndex[actualIndex.Name()] = actualIndex
		allActualIndexNames = append(allActualIndexNames, actualIndex.Name())
	}
	for _, expIndex := range expIndexes {
		_, has := mNameActualIndex[expIndex.Name()]
		require.Truef(t, has, "expIndex name %s not in actualIndexes %v", expIndex.Name(), allActualIndexNames)
		require.Equal(t, mNameActualIndex[expIndex.Name()].IndexType(), expIndex.IndexType())
		require.Equal(t, mNameActualIndex[expIndex.Name()].Params(), expIndex.Params())
	}
}

// EqualColumn assert field data is equal of two columns
func EqualColumn(t *testing.T, columnA entity.Column, columnB entity.Column) {
	require.Equal(t, columnA.Name(), columnB.Name())
	require.Equal(t, columnA.Type(), columnB.Type())
	switch columnA.Type() {
	case entity.FieldTypeBool:
		require.ElementsMatch(t, columnA.(*entity.ColumnBool).Data(), columnB.(*entity.ColumnBool).Data())
	case entity.FieldTypeInt8:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt8).Data(), columnB.(*entity.ColumnInt8).Data())
	case entity.FieldTypeInt16:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt16).Data(), columnB.(*entity.ColumnInt16).Data())
	case entity.FieldTypeInt32:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt32).Data(), columnB.(*entity.ColumnInt32).Data())
	case entity.FieldTypeInt64:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt64).Data(), columnB.(*entity.ColumnInt64).Data())
	case entity.FieldTypeFloat:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloat).Data(), columnB.(*entity.ColumnFloat).Data())
	case entity.FieldTypeDouble:
		require.ElementsMatch(t, columnA.(*entity.ColumnDouble).Data(), columnB.(*entity.ColumnDouble).Data())
	case entity.FieldTypeVarChar:
		require.ElementsMatch(t, columnA.(*entity.ColumnVarChar).Data(), columnB.(*entity.ColumnVarChar).Data())
	case entity.FieldTypeJSON:
		log.Printf("columnA: %s", columnA.FieldData())
		log.Printf("columnB: %s", columnB.FieldData())
		require.Equal(t, reflect.TypeOf(columnA), reflect.TypeOf(columnB))
		switch columnA.(type) {
		case *entity.ColumnDynamic:
			require.ElementsMatch(t, columnA.(*entity.ColumnDynamic).Data(), columnB.(*entity.ColumnDynamic).Data())
		case *entity.ColumnJSONBytes:
			require.ElementsMatch(t, columnA.(*entity.ColumnJSONBytes).Data(), columnB.(*entity.ColumnJSONBytes).Data())
		}
	case entity.FieldTypeFloatVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloatVector).Data(), columnB.(*entity.ColumnFloatVector).Data())
	case entity.FieldTypeBinaryVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnBinaryVector).Data(), columnB.(*entity.ColumnBinaryVector).Data())
	case entity.FieldTypeFloat16Vector:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloat16Vector).Data(), columnB.(*entity.ColumnFloat16Vector).Data())
	case entity.FieldTypeBFloat16Vector:
		require.ElementsMatch(t, columnA.(*entity.ColumnBFloat16Vector).Data(), columnB.(*entity.ColumnBFloat16Vector).Data())
	case entity.FieldTypeSparseVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnSparseFloatVector).Data(), columnB.(*entity.ColumnSparseFloatVector).Data())
	case entity.FieldTypeArray:
		EqualArrayColumn(t, columnA, columnB)
	default:
		log.Printf("The column type not in: [%v, %v, %v,  %v, %v,  %v, %v,  %v, %v,  %v, %v, %v]",
			entity.FieldTypeBool, entity.FieldTypeInt8, entity.FieldTypeInt16, entity.FieldTypeInt32,
			entity.FieldTypeInt64, entity.FieldTypeFloat, entity.FieldTypeDouble, entity.FieldTypeString,
			entity.FieldTypeVarChar, entity.FieldTypeArray, entity.FieldTypeFloatVector, entity.FieldTypeBinaryVector)
	}
}

// EqualColumn assert field data is equal of two columns
func EqualArrayColumn(t *testing.T, columnA entity.Column, columnB entity.Column) {
	require.Equal(t, columnA.Name(), columnB.Name())
	require.IsType(t, columnA.Type(), entity.FieldTypeArray)
	require.IsType(t, columnB.Type(), entity.FieldTypeArray)
	switch columnA.(type) {
	case *entity.ColumnBoolArray:
		require.ElementsMatch(t, columnA.(*entity.ColumnBoolArray).Data(), columnB.(*entity.ColumnBoolArray).Data())
	case *entity.ColumnInt8Array:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt8Array).Data(), columnB.(*entity.ColumnInt8Array).Data())
	case *entity.ColumnInt16Array:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt16Array).Data(), columnB.(*entity.ColumnInt16Array).Data())
	case *entity.ColumnInt32Array:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt32Array).Data(), columnB.(*entity.ColumnInt32Array).Data())
	case *entity.ColumnInt64Array:
		require.ElementsMatch(t, columnA.(*entity.ColumnInt64Array).Data(), columnB.(*entity.ColumnInt64Array).Data())
	case *entity.ColumnFloatArray:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloatArray).Data(), columnB.(*entity.ColumnFloatArray).Data())
	case *entity.ColumnDoubleArray:
		require.ElementsMatch(t, columnA.(*entity.ColumnDoubleArray).Data(), columnB.(*entity.ColumnDoubleArray).Data())
	case *entity.ColumnVarCharArray:
		require.ElementsMatch(t, columnA.(*entity.ColumnVarCharArray).Data(), columnB.(*entity.ColumnVarCharArray).Data())
	default:
		log.Printf("Now support array type: [%v, %v, %v,  %v, %v,  %v, %v,  %v]",
			entity.FieldTypeBool, entity.FieldTypeInt8, entity.FieldTypeInt16, entity.FieldTypeInt32,
			entity.FieldTypeInt64, entity.FieldTypeFloat, entity.FieldTypeDouble, entity.FieldTypeVarChar)
	}
}

// CheckQueryResult check query result, column name, type and field
func CheckQueryResult(t *testing.T, actualColumns []entity.Column, expColumns []entity.Column) {
	require.GreaterOrEqual(t, len(actualColumns), len(expColumns),
		"The len of actual columns %d should greater or equal to the expected columns %d", len(actualColumns), len(expColumns))
	for _, expColumn := range expColumns {
		for _, actualColumn := range actualColumns {
			if expColumn.Name() == actualColumn.Name() {
				EqualColumn(t, expColumn, actualColumn)
			}
		}
	}
}

// CheckOutputFields check query output fields
func CheckOutputFields(t *testing.T, actualColumns []entity.Column, expFields []string) {
	actualFields := make([]string, 0)
	for _, actualColumn := range actualColumns {
		actualFields = append(actualFields, actualColumn.Name())
	}
	require.ElementsMatchf(t, expFields, actualFields, fmt.Sprintf("Expected search output fields: %v, actual: %v", expFields, actualFields))
}

// CheckSearchResult check search result, check nq, topk, ids, score
func CheckSearchResult(t *testing.T, actualSearchResults []client.SearchResult, expNq int, expTopK int) {
	require.Equal(t, len(actualSearchResults), expNq)
	for _, actualSearchResult := range actualSearchResults {
		require.Equal(t, actualSearchResult.ResultCount, expTopK)
	}
	// expContainedIds entity.Column
}

func EqualIntSlice(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type CheckIteratorOption func(opt *checkIteratorOpt)

type checkIteratorOpt struct {
	expBatchSize    []int
	expOutputFields []string
}

func WithExpBatchSize(expBatchSize []int) CheckIteratorOption {
	return func(opt *checkIteratorOpt) {
		opt.expBatchSize = expBatchSize
	}
}

func WithExpOutputFields(expOutputFields []string) CheckIteratorOption {
	return func(opt *checkIteratorOpt) {
		opt.expOutputFields = expOutputFields
	}
}

// check queryIterator: result limit, each batch size, output fields
func CheckQueryIteratorResult(ctx context.Context, t *testing.T, itr *client.QueryIterator, expLimit int, opts ...CheckIteratorOption) {
	opt := &checkIteratorOpt{}
	for _, o := range opts {
		o(opt)
	}
	actualLimit := 0
	var actualBatchSize []int
	for {
		rs, err := itr.Next(ctx)
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalf("QueryIterator next gets error: %v", err)
		}

		if opt.expBatchSize != nil {
			actualBatchSize = append(actualBatchSize, rs.Len())
		}
		var actualOutputFields []string
		if opt.expOutputFields != nil {
			for _, column := range rs {
				actualOutputFields = append(actualOutputFields, column.Name())
			}
			require.ElementsMatch(t, opt.expOutputFields, actualOutputFields)
		}
		actualLimit = actualLimit + rs.Len()
	}
	require.Equal(t, expLimit, actualLimit)
	if opt.expBatchSize != nil {
		log.Printf("QueryIterator result len: %v", actualBatchSize)
		require.True(t, EqualIntSlice(opt.expBatchSize, actualBatchSize))
	}
}

// CheckPersistentSegments check persistent segments
func CheckPersistentSegments(t *testing.T, actualSegments []*entity.Segment, expNb int64) {
	actualNb := int64(0)
	for _, segment := range actualSegments {
		actualNb = segment.NumRows + actualNb
	}
	require.Equal(t, actualNb, expNb)
}

func CheckTransfer(t *testing.T, actualRgs []*entity.ResourceGroupTransfer, expRgs []*entity.ResourceGroupTransfer) {
	if len(expRgs) == 0 {
		require.Len(t, actualRgs, 0)
	} else {
		_expRgs := make([]string, 0, len(expRgs))
		_actualRgs := make([]string, 0, len(actualRgs))
		for _, rg := range expRgs {
			_expRgs = append(_expRgs, rg.ResourceGroup)
		}
		for _, rg := range actualRgs {
			_actualRgs = append(_actualRgs, rg.ResourceGroup)
		}
		require.ElementsMatch(t, _expRgs, _actualRgs)
	}

}

func checkResourceGroupConfig(t *testing.T, actualConfig *entity.ResourceGroupConfig, expConfig *entity.ResourceGroupConfig) {
	if expConfig.Requests != nil {
		require.EqualValuesf(t, expConfig.Requests.NodeNum, actualConfig.Requests.NodeNum, "Requests.NodeNum mismatch")
	}

	if expConfig.Limits != nil {
		require.EqualValuesf(t, expConfig.Limits.NodeNum, actualConfig.Limits.NodeNum, "Limits.NodeNum mismatch")
	}

	if expConfig.TransferFrom != nil {
		CheckTransfer(t, expConfig.TransferFrom, actualConfig.TransferFrom)
	}

	if expConfig.TransferTo != nil {
		CheckTransfer(t, expConfig.TransferTo, actualConfig.TransferTo)
	}
}

func CheckResourceGroup(t *testing.T, actualRg *entity.ResourceGroup, expRg *entity.ResourceGroup) {
	require.EqualValues(t, expRg.Name, actualRg.Name, "ResourceGroup name mismatch")
	require.EqualValues(t, expRg.Capacity, actualRg.Capacity, "ResourceGroup capacity mismatch")
	if expRg.AvailableNodesNumber >= 0 {
		require.EqualValues(t, expRg.AvailableNodesNumber, len(actualRg.Nodes), "AvailableNodesNumber mismatch")
	}

	if expRg.Config != nil {
		checkResourceGroupConfig(t, actualRg.Config, expRg.Config)
	}

	if expRg.Nodes != nil {
		require.EqualValues(t, len(expRg.Nodes), len(actualRg.Nodes), "Nodes count mismatch")
	}
}

func getDbNames(dbs []entity.Database) []string {
	allDbNames := make([]string, 0, len(dbs))
	for _, db := range dbs {
		allDbNames = append(allDbNames, db.Name)
	}
	return allDbNames
}

// CheckContainsDb check collections contains collName
func CheckContainsDb(t *testing.T, dbs []entity.Database, dbName string) {
	allDbNames := getDbNames(dbs)
	require.Containsf(t, allDbNames, dbName, fmt.Sprintf("%s db not in dbs: %v", dbName, dbs))
}

// CheckNotContainsDb check collections contains collName
func CheckNotContainsDb(t *testing.T, dbs []entity.Database, dbName string) {
	allDbNames := getDbNames(dbs)
	require.NotContainsf(t, allDbNames, dbName, fmt.Sprintf("%s db should not be in dbs: %v", dbName, dbs))
}

// Checks whether the list contains the specified value
func CheckContainsValue(fieldNames []interface{}, fieldName interface{}) bool {
	for _, v := range fieldNames {
		if v == fieldName {
			return true
		}
	}
	return false
}
