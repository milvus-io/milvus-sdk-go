//go:build L0

package testcases

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test create default floatVec and binaryVec collection
func TestCreateCollection(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// prepare
	defaultFields := [][]*entity.Field{
		common.GenDefaultFields(false),
		common.GenDefaultBinaryFields(false, common.DefaultDimStr),
		common.GenDefaultVarcharFields(false),
	}
	for _, fields := range defaultFields {
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, false, fields)

		// create collection
		errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreateCollection, true)

		// check describe collection
		collection, _ := mc.DescribeCollection(ctx, collName)
		common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)

		// check collName in ListCollections
		collections, errListCollection := mc.ListCollections(ctx)
		common.CheckErr(t, errListCollection, true)
		common.CheckContainsCollection(t, collections, collName)
	}
}

func TestCreateAutoIdCollection(t *testing.T) {
	t.Skipf("issue: %v", "https://github.com/milvus-io/milvus-sdk-go/issues/343")
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// prepare
	defaultFields := [][]*entity.Field{
		common.GenDefaultFields(true),
		common.GenDefaultBinaryFields(true, common.DefaultDimStr),
		common.GenDefaultVarcharFields(true),
	}
	for _, fields := range defaultFields {
		collName := common.GenRandomString(6)
		schema := common.GenSchema(collName, true, fields)

		// create collection
		errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreateCollection, true)

		// check describe collection
		collection, _ := mc.DescribeCollection(ctx, collName)
		log.Printf("collection schema autoid: %v", collection.Schema.AutoID)
		log.Printf("collection pk field autoid: %v", collection.Schema.Fields[0].AutoID)
		common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)

		// check collName in ListCollections
		collections, errListCollection := mc.ListCollections(ctx)
		common.CheckErr(t, errListCollection, true)
		common.CheckContainsCollection(t, collections, collName)
	}
}

// test create collection with invalid collection and field name
func TestCreateCollectionInvalidName(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := common.GenDefaultFields(false)
	type invalidNameStruct struct {
		name   string
		errMsg string
	}

	invalidNames := []invalidNameStruct{
		{name: "", errMsg: "not be empty"},
		{name: "12-s", errMsg: "name must be an underscore or letter"},
		{name: "(mn)", errMsg: "name must be an underscore or letter"},
		{name: "中文", errMsg: "name must be an underscore or letter"},
		{name: "%$#", errMsg: "name must be an underscore or letter"},
		{name: common.GenLongString(common.MaxCollectionNameLen + 1), errMsg: "name must be less than 255 characters"},
	}

	for _, invalidName := range invalidNames {
		schema := &entity.Schema{
			CollectionName: invalidName.name,
			AutoID:         false,
			Fields:         fields,
		}
		errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreateCollection, false, invalidName.errMsg)
	}

	for _, invalidName := range invalidNames {
		invalidField := []*entity.Field{
			common.GenScalaField(invalidName.name, entity.FieldTypeInt64, true, false),
			common.GenVectorField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.DefaultDimStr),
		}
		schema := &entity.Schema{
			CollectionName: common.GenRandomString(6),
			AutoID:         false,
			Fields:         invalidField,
		}
		errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreateCollection, false, invalidName.errMsg)
	}
}

// test create collection with nil fields and nil schema
func TestCreateCollectionWithNil(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// create collection with nil schema
	errCreateCollection := mc.CreateCollection(ctx, nil, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, false, "nil schema")

	// create collection with nil fields
	schema := common.GenSchema(common.GenRandomString(6), true, nil)
	errCreateCollection2 := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection2, false, "vector field not set")
}

// test create collection with invalid fields: without pk, without vec field, multi pk field, multi vector field
func TestCreateCollectionInvalidFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	type invalidFieldsStruct struct {
		fields []*entity.Field
		errMsg string
	}
	invalidFields := []invalidFieldsStruct{
		// // create collection without pk field
		{fields: []*entity.Field{common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr)},
			errMsg: "primary key is not specified"},

		// create collection without vector field
		{fields: []*entity.Field{common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, false)},
			errMsg: "vector field not set"},

		// create collection with multi pk fields
		{fields: []*entity.Field{
			common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, false),
			common.GenScalaField(common.DefaultFloatVecFieldName, entity.FieldTypeInt64, true, true),
			common.GenVectorField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.DefaultDimStr),
		}, errMsg: "only one primary key only"},

		// create collection with multi vector fields
		{fields: []*entity.Field{
			common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, false),
			common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
			common.GenVectorField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.DefaultDimStr),
		}, errMsg: "multiple vector fields is not supported"},
	}

	for _, invalidField := range invalidFields {
		schema := common.GenSchema(common.GenRandomString(6), false, invalidField.fields)
		errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errWithoutPk, false, invalidField.errMsg)
	}
}

func TestCreateCollectionNonInt64AutoField(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	invalidPkFields := []entity.FieldType{
		entity.FieldTypeBool,
		entity.FieldTypeInt8,
		entity.FieldTypeInt16,
		entity.FieldTypeInt32,
		entity.FieldTypeFloat,
		entity.FieldTypeDouble,
		entity.FieldTypeVarChar,
		entity.FieldTypeString,
		entity.FieldTypeNone,
	}
	for _, fieldType := range invalidPkFields {
		fields := []*entity.Field{
			common.GenScalaField("non-auto-id", fieldType, false, true),
			common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, true),
			common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
		}
		schema := common.GenSchema(common.GenRandomString(6), true, fields)
		errNonInt64Field := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errNonInt64Field, false, "only int64 column can be auto generated id")
	}
}

// test create collection with duplicate field name
func TestCreateCollectionDuplicateField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// duplicate field
	fields := []*entity.Field{
		common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, false),
		common.GenScalaField(common.DefaultFloatFieldName, entity.FieldTypeFloat, false, false),
		common.GenScalaField(common.DefaultFloatFieldName, entity.FieldTypeFloat, false, false),
		common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
	}

	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, fields)
	errDupField := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errDupField, false, "duplicated field name")
}

// test create collection with invalid pk field type
func TestCreateCollectionInvalidPkType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	invalidPkFields := []entity.FieldType{
		entity.FieldTypeBool,
		entity.FieldTypeInt8,
		entity.FieldTypeInt16,
		entity.FieldTypeInt32,
		entity.FieldTypeFloat,
		entity.FieldTypeDouble,
		entity.FieldTypeString,
		entity.FieldTypeNone,
	}
	for _, fieldType := range invalidPkFields {
		fields := []*entity.Field{
			common.GenScalaField("invalid", fieldType, true, false),
			common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
		}
		schema := common.GenSchema(common.GenRandomString(6), false, fields)
		errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errWithoutPk, false, "only int64 and varchar column can be primary key for now")
	}
}

// test create collection with multi auto id
func TestCreateCollectionMultiAutoId(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := []*entity.Field{
		common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, true),
		common.GenScalaField("dupInt", entity.FieldTypeInt64, false, true),
		common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
	}
	schema := common.GenSchema(common.GenRandomString(6), false, fields)
	errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errWithoutPk, false, "only one auto id is available")
}

// test create collection with different autoId between pk field and schema
func TestCreateCollectionInconsistentAutoId(t *testing.T) {
	t.Skipf("Issue: %s", "https://github.com/milvus-io/milvus-sdk-go/issues/342")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := []*entity.Field{
		// autoId true
		common.GenScalaField(common.DefaultIntFieldName, entity.FieldTypeInt64, true, true),
		common.GenVectorField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.DefaultDimStr),
	}
	// autoId false
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, fields)
	errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errWithoutPk, true, "only one auto id is available")
	collection, _ := mc.DescribeCollection(ctx, collName)
	log.Printf("collection schema AutoId is %v)", collection.Schema.AutoID)
	log.Printf("collection pk field AutoId is %v)", collection.Schema.Fields[0].AutoID)
}

// test create collection with field description and schema description
func TestCreateCollectionDescription(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	var fields = []*entity.Field{
		{
			Name:        common.DefaultIntFieldName,
			DataType:    entity.FieldTypeInt64,
			PrimaryKey:  true,
			Description: "pk field",
		},
		{
			Name:       common.DefaultFloatVecFieldName,
			DataType:   entity.FieldTypeFloatVector,
			TypeParams: map[string]string{entity.TypeParamDim: common.DefaultDimStr},
		},
	}
	schema := &entity.Schema{
		CollectionName: common.GenRandomString(6),
		AutoID:         false,
		Fields:         fields,
		Description:    "schema",
	}
	errCreate := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreate, true)

	collection, _ := mc.DescribeCollection(ctx, schema.CollectionName)
	require.Equal(t, collection.Schema.Description, "schema")
	require.Equal(t, collection.Schema.Fields[0].Description, "pk field")
}

// test create collection with invalid dim
func TestCreateCollectionInvalidDim(t *testing.T) {
	t.Parallel()
	type invalidDimStruct struct {
		dim    string
		errMsg string
	}
	invalidDims := []invalidDimStruct{
		{dim: "10", errMsg: "should be multiple of 8"},
		{dim: "0", errMsg: "should be in range 1 ~ 32768"},
		{dim: "", errMsg: "invalid syntax"},
		{dim: "中文", errMsg: "invalid syntax"},
		{dim: "%$#", errMsg: "invalid syntax"},
		{dim: fmt.Sprintf("%d", common.MaxDim+1), errMsg: "should be in range 1 ~ 32768"},
	}

	// connect
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// create binary collection with autoID true
	for _, invalidDim := range invalidDims {
		collName := common.GenRandomString(6)
		binaryFields := common.GenDefaultBinaryFields(true, invalidDim.dim)
		schema := common.GenSchema(collName, true, binaryFields)
		errCreate := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreate, false, invalidDim.errMsg)
	}
}

// -- Get Collection Statistics --

func TestGetStaticsCollectionNotExisted(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	// flush and check row count
	_, errStatist := mc.GetCollectionStatistics(ctx, "collName")
	common.CheckErr(t, errStatist, false, "collection collName does not exist")
}
