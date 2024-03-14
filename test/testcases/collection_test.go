//go:build L0

package testcases

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

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
		common.GenDefaultBinaryFields(false, common.DefaultDim),
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
		common.GenDefaultBinaryFields(true, common.DefaultDim),
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
		field := common.GenField(invalidName.name, entity.FieldTypeInt64, common.WithIsPrimaryKey(true))
		if invalidName.name == "" {
			field.WithName("")
		}
		invalidField := []*entity.Field{
			field,
			common.GenField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.WithDim(common.DefaultDim)),
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

// test create collection with invalid fields: without pk, without vec field, multi pk field
// TODO multi vector field
func TestCreateCollectionInvalidFields(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	type invalidFieldsStruct struct {
		fields []*entity.Field
		errMsg string
	}
	invalidFields := []invalidFieldsStruct{
		// create collection without pk field
		{fields: []*entity.Field{common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim))},
			errMsg: "primary key is not specified"},

		// create collection without vector field
		{fields: []*entity.Field{common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true))},
			errMsg: "vector field not set"},

		// create collection with multi pk fields
		{fields: []*entity.Field{
			common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(true)),
			common.GenField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.WithDim(common.DefaultDim)),
		}, errMsg: "only one primary key only"},

		// create collection with multi vector fields
		//{fields: []*entity.Field{
		//	common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
		//	common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		//	common.GenField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.WithDim(common.DefaultDim)),
		//}, errMsg: "multiple vector fields is not supported"},

		// create collection with None field type
		{fields: []*entity.Field{
			common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
			common.GenField("", entity.FieldTypeNone),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}, errMsg: "data type None is not valid"},

		// create collection with String field type
		{fields: []*entity.Field{
			common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
			common.GenField("", entity.FieldTypeString),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}, errMsg: "string data type not supported yet, please use VarChar type instead"},

		// varchar field not specify max_length
		{fields: []*entity.Field{
			common.GenField(common.DefaultVarcharFieldName, entity.FieldTypeVarChar, common.WithIsPrimaryKey(true)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}, errMsg: "type param(max_length) should be specified for varChar field"},
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
		// entity.FieldTypeVarChar,
		entity.FieldTypeString,
		entity.FieldTypeNone,
		entity.FieldTypeJSON,
	}
	for _, fieldType := range invalidPkFields {
		fields := []*entity.Field{
			common.GenField(common.DefaultIntFieldName, fieldType, common.WithIsPrimaryKey(true), common.WithAutoID(true)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}
		schema := common.GenSchema(common.GenRandomString(6), true, fields)
		errNonInt64Field := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errNonInt64Field, false, "only int64 and varchar column can be primary key for now")
	}
}

// test create collection with duplicate field name
func TestCreateCollectionDuplicateField(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// duplicate field
	fields := []*entity.Field{
		common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
		common.GenField(common.DefaultFloatFieldName, entity.FieldTypeFloat),
		common.GenField(common.DefaultFloatFieldName, entity.FieldTypeFloat),
		common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
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
		entity.FieldTypeJSON,
	}
	for _, fieldType := range invalidPkFields {
		fields := []*entity.Field{
			common.GenField("invalid", fieldType, common.WithIsPrimaryKey(true)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}
		schema := common.GenSchema(common.GenRandomString(6), false, fields)
		errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errWithoutPk, false, "only int64 and varchar column can be primary key for now")
	}
}

// test create collection with partition key not supported field type
func TestCreateCollectionInvalidPartitionKeyType(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	invalidPkFields := []entity.FieldType{
		entity.FieldTypeBool,
		entity.FieldTypeInt8,
		entity.FieldTypeInt16,
		entity.FieldTypeInt32,
		entity.FieldTypeFloat,
		entity.FieldTypeDouble,
		entity.FieldTypeJSON,
	}
	pkField := common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true))
	for _, fieldType := range invalidPkFields {
		fields := []*entity.Field{
			pkField,
			common.GenField("invalid", fieldType, common.WithIsPartitionKey(true)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
		}
		schema := common.GenSchema(common.GenRandomString(6), false, fields)
		errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errWithoutPk, false, "the data type of partition key should be Int64 or VarChar")
	}
}

// test create collection with multi auto id
func TestCreateCollectionMultiAutoId(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := []*entity.Field{
		common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(true)),
		common.GenField("dupInt", entity.FieldTypeInt64, common.WithAutoID(true)),
		common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
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
		common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(true)),
		common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim)),
	}
	// autoId false
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, fields)
	errWithoutPk := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errWithoutPk, true, "only one auto id is available")
	collection, _ := mc.DescribeCollection(ctx, collName)
	log.Printf("collection schema AutoID is %v)", collection.Schema.AutoID)
	log.Printf("collection pk field AutoID is %v)", collection.Schema.Fields[0].AutoID)
}

// test create collection with field description and schema description
func TestCreateCollectionDescription(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// gen field with description
	pkField := common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true),
		common.WithFieldDescription("pk field"))
	vecField := common.GenField("", entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim))
	var fields = []*entity.Field{
		pkField, vecField,
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
func TestCreateBinaryCollectionInvalidDim(t *testing.T) {
	t.Parallel()
	type invalidDimStruct struct {
		dim    string
		errMsg string
	}
	invalidDims := []invalidDimStruct{
		{dim: "10", errMsg: "should be multiple of 8"},
		{dim: "0", errMsg: "should be in range 2 ~ 32768"},
		{dim: "1", errMsg: "should be in range 2 ~ 32768"},
		{dim: "", errMsg: "invalid syntax"},
		{dim: "中文", errMsg: "invalid syntax"},
		{dim: "%$#", errMsg: "invalid syntax"},
		{dim: fmt.Sprintf("%d", common.MaxDim*9), errMsg: "binary vector dimension should be in range 2 ~ 262144"},
		{dim: fmt.Sprintf("%d", common.MaxDim*8+1), errMsg: "binary vector dimension should be multiple of 8"},
	}

	// connect
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// create binary collection with autoID true
	pkField := common.GenField("", entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(true))
	for _, invalidDim := range invalidDims {
		collName := common.GenRandomString(6)
		binaryFields := entity.NewField().
			WithName(common.DefaultFloatVecFieldName).
			WithDataType(entity.FieldTypeBinaryVector).
			WithTypeParams(entity.TypeParamDim, invalidDim.dim)
		schema := common.GenSchema(collName, true, []*entity.Field{pkField, binaryFields})
		errCreate := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreate, false, invalidDim.errMsg)
	}
}

func TestCreateFloatCollectionInvalidDim(t *testing.T) {
	t.Parallel()
	type invalidDimStruct struct {
		dim    string
		errMsg string
	}
	invalidDims := []invalidDimStruct{
		{dim: "0", errMsg: "should be in range 2 ~ 32768"},
		{dim: "1", errMsg: "should be in range 2 ~ 32768"},
		{dim: "", errMsg: "invalid syntax"},
		{dim: "中文", errMsg: "invalid syntax"},
		{dim: "%$#", errMsg: "invalid syntax"},
		{dim: fmt.Sprintf("%d", common.MaxDim+1), errMsg: "float vector dimension should be in range 2 ~ 32768"},
	}

	// connect
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// create binary collection with autoID true
	pkField := common.GenField("", entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(true))
	for _, invalidDim := range invalidDims {
		collName := common.GenRandomString(6)
		binaryFields := entity.NewField().
			WithName(common.DefaultFloatVecFieldName).
			WithDataType(entity.FieldTypeFloatVector).
			WithTypeParams(entity.TypeParamDim, invalidDim.dim)
		schema := common.GenSchema(collName, true, []*entity.Field{pkField, binaryFields})
		errCreate := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, errCreate, false, invalidDim.errMsg)
	}
}

func TestCreateJsonCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// fields
	fields := common.GenDefaultFields(true)
	for i := 0; i < 2; i++ {
		jsonField1 := common.GenField("", entity.FieldTypeJSON)
		fields = append(fields, jsonField1)
	}

	// schema
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, fields)

	// create collection
	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, true)

	// check describe collection
	collection, _ := mc.DescribeCollection(ctx, collName)
	common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)

	// check collName in ListCollections
	collections, errListCollection := mc.ListCollections(ctx)
	common.CheckErr(t, errListCollection, true)
	common.CheckContainsCollection(t, collections, collName)
}

// test create collection enable dynamic field
func TestCreateCollectionDynamicSchema(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, common.GenDefaultFields(false), common.WithEnableDynamicField(true))

	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, true)

	// check describe collection
	collection, _ := mc.DescribeCollection(ctx, collName)
	common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)
	require.Truef(t, collection.Schema.EnableDynamicField, "Expected collection.Schema.EnableDynamicField is True")

	// check collName in ListCollections
	collections, errListCollection := mc.ListCollections(ctx)
	common.CheckErr(t, errListCollection, true)
	common.CheckContainsCollection(t, collections, collName)
}

// test create collection enable dynamic field by collection opt
func TestCreateCollectionDynamic(t *testing.T) {
	t.Skip("Waiting for congqi to update schema.EnableDynamicField according to the CreateCollectionOption.EnableDynamicSchema")
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, common.GenDefaultFields(false))

	err := mc.CreateCollection(ctx, schema, common.DefaultShards, client.WithEnableDynamicSchema(true))
	common.CheckErr(t, err, true)

	// check describe collection
	collection, _ := mc.DescribeCollection(ctx, collName)
	common.CheckCollection(t, collection, collName, common.DefaultShards, schema, common.DefaultConsistencyLevel)
	require.Truef(t, collection.Schema.EnableDynamicField, "Expected collection.Schema.EnableDynamicField is True")

	// check collName in ListCollections
	collections, errListCollection := mc.ListCollections(ctx)
	common.CheckErr(t, errListCollection, true)
	common.CheckContainsCollection(t, collections, collName)
}

// test create collection contains field name: $meta -> error
func TestCreateCollectionFieldMeta(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	collName := common.GenRandomString(6)
	fields := common.GenDefaultFields(true)
	fields = append(fields, common.GenField(common.DefaultDynamicFieldName, entity.FieldTypeJSON))
	schema := common.GenSchema(collName, false, fields, common.WithEnableDynamicField(true))

	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, false, fmt.Sprintf("Invalid field name: %s. The first character of a field name must be an underscore or letter", common.DefaultDynamicFieldName))
}

func TestCreateArrayFieldInvalidCapacity(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := common.GenDefaultFields(true)
	// array field no Capacity
	arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(entity.FieldTypeFloat))
	schema := common.GenSchema(common.GenRandomString(6), false, append(fields, arrayField), common.WithEnableDynamicField(true))
	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, false, "type param(max_capacity) should be specified for array field")

	// invalid Capacity
	for _, invalidCapacity := range []int64{-1, 0, common.MaxCapacity + 1} {
		arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(entity.FieldTypeFloat), common.WithMaxCapacity(invalidCapacity))
		schema := common.GenSchema(common.GenRandomString(6), false, append(fields, arrayField), common.WithEnableDynamicField(true))
		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, false, "the maximum capacity specified for a Array should be in (0, 4096]")
	}
}

// test create collection varchar array with invalid max length
func TestCreateVarcharArrayInvalidLength(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	fields := common.GenDefaultFields(true)
	varcharArrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(entity.FieldTypeVarChar), common.WithMaxCapacity(100))
	schema := common.GenSchema(common.GenRandomString(6), false, append(fields, varcharArrayField), common.WithEnableDynamicField(true))
	err := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, err, false, "type param(max_length) should be specified for varChar field")

	// invalid max length
	for _, invalidLength := range []int64{-1, 0, common.MaxLength + 1} {
		arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(entity.FieldTypeVarChar),
			common.WithMaxCapacity(100), common.WithMaxLength(invalidLength))
		schema := common.GenSchema(common.GenRandomString(6), false, append(fields, arrayField), common.WithEnableDynamicField(true))
		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, false, "the maximum length specified for a VarChar should be in (0, 65535]: invalid parameter")
	}
}

// test create collection array field not supported type
func TestCreateArrayNotSupportedFieldType(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// not supported ElementType: Array, Json, FloatVector, BinaryVector
	fields := common.GenDefaultFields(true)
	for _, fieldType := range []entity.FieldType{entity.FieldTypeArray, entity.FieldTypeJSON, entity.FieldTypeBinaryVector, entity.FieldTypeFloatVector} {
		arrayField := common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(fieldType), common.WithMaxCapacity(100))

		schema := common.GenSchema(common.GenRandomString(6), false, append(fields, arrayField), common.WithEnableDynamicField(true))

		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, false, fmt.Sprintf("element type %s is not supported", fieldType.Name()))
	}

	// NoneType ElementType
	noneArrayFields := []*entity.Field{
		common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithElementType(entity.FieldTypeNone), common.WithMaxCapacity(100)),
		common.GenField(common.DefaultArrayFieldName, entity.FieldTypeArray, common.WithMaxCapacity(100)),
	}
	for _, noneArrayField := range noneArrayFields {
		schema := common.GenSchema(common.GenRandomString(6), false, append(fields, noneArrayField), common.WithEnableDynamicField(true))
		err := mc.CreateCollection(ctx, schema, common.DefaultShards)
		common.CheckErr(t, err, false, "element data type None is not valid")
	}
}

func TestCreateCollectionAllFields(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	allFields := common.GenAllFields()
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, allFields)

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)
}

// the num of vector fields > default limit=4
func TestCreateMultiVectorExceed(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	fields := []*entity.Field{
		common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true)),
	}
	for i := 0; i < common.MaxVectorFieldNum+1; i++ {
		field := common.GenField(fmt.Sprintf("%s-%d", common.DefaultFloatVecFieldName, i), entity.FieldTypeFloatVector, common.WithDim(128))
		fields = append(fields, field)
	}
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, fields)

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, false, "maximum vector field's number should be limited to 4")
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
