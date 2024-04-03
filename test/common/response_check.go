package common

import (
	"fmt"
	"log"
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
				t.FailNow()
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
	//require.Equal(t, fieldA.IndexParams, fieldB.IndexParams)
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
	expSchema *entity.Schema, expConsistencyLevel entity.ConsistencyLevel) {
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
func CheckInsertResult(t *testing.T, actualIds entity.Column, expIds entity.Column) {
	require.Equal(t, actualIds.Len(), expIds.Len())
	switch expIds.Type() {
	// pk field support int64 and varchar type
	case entity.FieldTypeInt64:
		require.ElementsMatch(t, actualIds.(*entity.ColumnInt64).Data(), expIds.(*entity.ColumnInt64).Data())
	case entity.FieldTypeVarChar:
		require.ElementsMatch(t, actualIds.(*entity.ColumnVarChar).Data(), expIds.(*entity.ColumnVarChar).Data())
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
		log.Printf("columnA: %s", columnA.(*entity.ColumnJSONBytes).Data())
		log.Printf("columnB: %s", columnB.(*entity.ColumnJSONBytes).Data())
		require.ElementsMatch(t, columnA.(*entity.ColumnJSONBytes).Data(), columnB.(*entity.ColumnJSONBytes).Data())
	case entity.FieldTypeFloatVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloatVector).Data(), columnB.(*entity.ColumnFloatVector).Data())
	case entity.FieldTypeBinaryVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnBinaryVector).Data(), columnB.(*entity.ColumnBinaryVector).Data())
	case entity.FieldTypeArray:
		log.Println("TODO support column element type")
	default:
		log.Printf("The column type not in: [%v, %v, %v,  %v, %v,  %v, %v,  %v, %v,  %v, %v, %v]",
			entity.FieldTypeBool, entity.FieldTypeInt8, entity.FieldTypeInt16, entity.FieldTypeInt32,
			entity.FieldTypeInt64, entity.FieldTypeFloat, entity.FieldTypeDouble, entity.FieldTypeString,
			entity.FieldTypeVarChar, entity.FieldTypeArray, entity.FieldTypeFloatVector, entity.FieldTypeBinaryVector)

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
	log.Printf("actual fields: %v", actualFields)
	log.Printf("expected fields: %v", expFields)
	require.ElementsMatchf(t, expFields, actualFields, fmt.Sprintf("Expected search output fields: %v, actual: %v", expFields, actualFields))
}

// CheckSearchResult check search result, check nq, topk, ids, score
func CheckSearchResult(t *testing.T, actualSearchResults []client.SearchResult, expNq int, expTopK int) {
	require.Equal(t, len(actualSearchResults), expNq)
	for _, actualSearchResult := range actualSearchResults {
		require.Equal(t, actualSearchResult.ResultCount, expTopK)
	}
	//expContainedIds entity.Column

}

// CheckPersistentSegments check persistent segments
func CheckPersistentSegments(t *testing.T, actualSegments []*entity.Segment, expNb int64) {
	actualNb := int64(0)
	for _, segment := range actualSegments {
		actualNb = segment.NumRows + actualNb
	}
	require.Equal(t, actualNb, expNb)
}

func CheckResourceGroup(t *testing.T, actualRg *entity.ResourceGroup, expRg *entity.ResourceGroup) {
	require.EqualValues(t, expRg, actualRg)
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
