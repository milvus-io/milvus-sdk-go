package common

import (
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"
	"log"
	"testing"
)

// check err and errMsg
func CheckErr(t *testing.T, actualErr error, expErrNil bool, expErrorMsg string)  {
	if expErrNil {
		require.NoError(t, actualErr)
	} else {
		require.ErrorContains(t, actualErr, expErrorMsg)
	}
}

// equal two fields
func EqualFields(t *testing.T, fieldA *entity.Field, fieldB *entity.Field) {
	require.Equal(t, fieldA.Name, fieldB.Name)
	require.Equal(t, fieldA.AutoID, fieldB.AutoID)
	require.Equal(t, fieldA.PrimaryKey, fieldB.PrimaryKey)
	require.Equal(t, fieldA.Description, fieldB.Description)
	require.Equal(t, fieldA.DataType, fieldB.DataType)
	// TODO default TypeParams and IndexParams
	//Expected :map[string]string(nil)
	//Actual   :map[string]string{}

	// check vector field dim
	switch fieldA.DataType {
	case entity.FieldTypeFloatVector:
		require.Equal(t, fieldA.TypeParams["dim"], fieldB.TypeParams["dim"])
	case entity.FieldTypeBinaryVector:
		require.Equal(t, fieldA.TypeParams["dim"], fieldB.TypeParams["dim"])
	}
	//require.Equal(t, fieldA.IndexParams, fieldB.IndexParams)
}

// equal two schemas
// TODO check
func EqualSchema(t *testing.T, schemaA entity.Schema, schemaB entity.Schema)  {
	require.Equal(t, schemaA.CollectionName, schemaB.CollectionName)
	require.Equal(t, schemaA.Description, schemaB.Description)
	//require.Equal(t, schemaA.AutoID, schemaB.AutoID)
	require.Equal(t, len(schemaA.Fields), len(schemaB.Fields))
	for i:=0; i< len(schemaA.Fields); i++ {
		EqualFields(t, schemaA.Fields[i], schemaB.Fields[i])
	}
}
// check
func CheckCollection(t *testing.T, actualCollection *entity.Collection, expCollName string, expShardNum int32,
	expSchema *entity.Schema, expConsistencyLevel entity.ConsistencyLevel) {
	require.Equal(t, expCollName, actualCollection.Name)
	require.Equal(t, expShardNum, actualCollection.ShardNum)
	require.Equal(t, expConsistencyLevel, actualCollection.ConsistencyLevel)
	EqualSchema(t, *expSchema, *actualCollection.Schema)
}

// check collections contains collName
func CheckContainsCollection(t *testing.T, collections []*entity.Collection, collName string)  {
	allCollNames := make([]string, 0, len(collections))
	for _, collection := range collections {
		allCollNames = append(allCollNames, collection.Name)
	}
	require.Contains(t, allCollNames, collName)
}

// check insert result, ids len (insert count), ids data (pks, but no auto ids)
func CheckInsertResult (t *testing.T, actualIds entity.Column, expIds entity.Column)  {
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

// check index result, index type, metric type, index params
//func CheckIndexResult(t *testing.T, actualIndexes []entity.Index, expIndexes []entity.Index)  {
//	for _, expIndex := range expIndexes {
//
//	}
//}


// assert field data is equal of two columns
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
	case entity.FieldTypeFloatVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnFloatVector).Data(), columnB.(*entity.ColumnFloatVector).Data())
	case entity.FieldTypeBinaryVector:
		require.ElementsMatch(t, columnA.(*entity.ColumnBinaryVector).Data(), columnB.(*entity.ColumnBinaryVector).Data())
	default:
		log.Printf("The column type not in: [%v, %v, %v,  %v, %v,  %v, %v,  %v, %v,  %v, %v]",
			entity.FieldTypeBool, entity.FieldTypeInt8, entity.FieldTypeInt16, entity.FieldTypeInt32,
			entity.FieldTypeInt64, entity.FieldTypeFloat, entity.FieldTypeDouble, entity.FieldTypeString,
			entity.FieldTypeVarChar, entity.FieldTypeFloatVector, entity.FieldTypeBinaryVector)


	}
}

// check query result, column name, type and field date
func CheckQueryResult(t *testing.T, actualColumns []entity.Column, expColumns []entity.Column) {
	if len(actualColumns) <= len(expColumns) {
		t.Fatalf("The len of actual columns should large or equal to the expcted columns")
	}
	for _, expColumn := range expColumns {
		for _, actualColumn := range actualColumns {
			if expColumn.Name() == actualColumn.Name() {
				EqualColumn(t, expColumn, actualColumn)
			}
		}
	}
}

// check query output fields
func CheckOutputFields(t *testing.T, actualColumns []entity.Column, expFields []string) {
	actualFields := make([]string, 0)
	for _, actualColumn := range actualColumns {
		actualFields = append(actualFields, actualColumn.Name())
	}
	require.ElementsMatch(t, actualFields, expFields)
}

// check search result, check nq, topk, ids, score
func CheckSearchResult(t *testing.T, actualSearchResults []client.SearchResult, expNq int64, expTopk int64)  {
	require.Equal(t, len(actualSearchResults), expNq)
	for _, actualSearchResult := range actualSearchResults {
		require.Equal(t, actualSearchResult.ResultCount, expTopk)
	}
}