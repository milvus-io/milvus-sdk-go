package tests

import (
	"context"
	"fmt"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	ut "github.com/milvus-io/milvus-sdk-go/v2/tests/testutil"
	"github.com/stretchr/testify/assert"
)

var nameTests = []struct {
	name string
}{
	{"123"},
	{"中文"},
	{"(mn)"},
	{"%$#"},
}

var dimTests = []struct {
	dim int64
}{
	{-1},
	{0},
	{32769},
}

// Test create collection with default schema
func TestClientCreateCollection(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	name := ut.GenRandomString(8)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	schema := ut.GenSchema(name, false, fields)
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	assert.NoError(t, err)
	isExist, err1 := c.HasCollection(ctx, name)
	assert.NoError(t, err1)
	assert.True(t, isExist)
}

// Test create collection without name, expected an error
func TestCollectionWithoutName(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	schema := &entity.Schema{
		AutoID: false,
		Fields: fields,
	}
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	errorStr := "collection name cannot be empty"
	assert.EqualError(t, err, errorStr)
}

// Test create collection without fields, expected an error
func TestCollectionWithoutFields(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	//fields := ut.GenDefaultFields()
	name := ut.GenRandomString(8)
	schema := &entity.Schema{
		CollectionName: name,
	}
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	errorStr := "vector field not set"
	assert.EqualError(t, err, errorStr)
}

// Test create collection with invalid name, expected an error
func TestCollectionInvalidName(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	schema := ut.GenSchema("name", false, fields)
	for _, test := range nameTests {
		schema.CollectionName = test.name
		err := c.CreateCollection(ctx, schema, ut.DefaultShards)

		// TODO Inaccurate error msg
		expError := "Invalid collection name"
		assert.Contains(t, err.Error(), expError)
	}
}

// TODO issue: #217
// Test create collection with invalid field name, expected an error
func TestCollectionInvalidFieldName(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	collectionName := ut.GenRandomString(8)
	for _, test := range nameTests {
		fields[0].Name = test.name
		t.Logf("field -- name: %s\n", fields[0].Name)
		schema := ut.GenSchema(collectionName, false, fields)
		err := c.CreateCollection(ctx, schema, ut.DefaultShards)
		t.Log(err)
	}
}

// TODO issue: #218
// Test create autoId collection
func TestCollectionAutoId(t *testing.T) {
	t.Skip("issue #218")

	ctx := context.Background()
	c := GenClient(t)
	name := ut.GenRandomString(8)
	t.Logf("exp name: %s", name)
	fields := ut.GenDefaultFields(ut.DefaultDim)
	schema := ut.GenSchema(name, true, fields)
	t.Logf("exp autoId: %t", schema.AutoID)
	c.CreateCollection(ctx, schema, ut.DefaultShards)
	collection, _ := c.DescribeCollection(ctx, name)
	t.Logf("actual name: %s", collection.Name)
	t.Logf("actual autoId: %t", collection.Schema.AutoID)
	assert.Equal(t, schema.AutoID, collection.Schema.AutoID)
}

// TODO issue: #219
// Test create collection only with vector field, expected an error
func TestCollectionOnlyVector(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	name := ut.GenRandomString(8)
	var fields = []*entity.Field{
		{
			Name:     name,
			DataType: entity.FieldTypeFloatVector,
			TypeParams: map[string]string{
				"dim": fmt.Sprintf("%d", ut.DefaultDim),
			},
		},
	}
	schema := &entity.Schema{
		CollectionName: name,
		Fields:         fields,
	}
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	t.Log(err)
	isExist, _ := c.HasCollection(ctx, name)
	t.Log(isExist)
}

// Test create collection without vector field, expected an error
func TestCollectionWithoutVector(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	fields := []*entity.Field{
		{
			Name:       ut.DefaultIntFieldName,
			DataType:   entity.FieldTypeInt64,
			PrimaryKey: true,
		},
	}
	name := ut.GenRandomString(8)
	schema := &entity.Schema{
		CollectionName: name,
		Fields:         fields,
	}
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	expError := "vector field not set"
	assert.Contains(t, err.Error(), expError)
}

// Test create collection with invalid dim
// TODO issue: #220
func TestCollectionInvalidDim(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	for _, test := range dimTests {
		name := ut.GenRandomString(8)
		fields := ut.GenDefaultFields(test.dim)
		t.Log(fields[2].TypeParams["dim"])
		schema := ut.GenSchema(name, false, fields)
		err := c.CreateCollection(ctx, schema, ut.DefaultDim)
		t.Log(err)
	}
}

// Test create collection with wrong keyword "dim", like "dims", expected an error
// TODO issue: #221
func TestCollectionWrongDimKey(t *testing.T) {
	ctx := context.Background()
	c := GenClient(t)
	name := ut.GenRandomString(8)
	fields := []*entity.Field{
		{
			Name:       ut.DefaultIntFieldName,
			DataType:   entity.FieldTypeInt64,
			PrimaryKey: true,
		},
		{
			Name:     ut.DefaultFloatVecFieldName,
			DataType: entity.FieldTypeFloatVector,
			TypeParams: map[string]string{
				"dims": fmt.Sprintf("%d", ut.DefaultDim),
			},
		},
	}
	schema := ut.GenSchema(name, false, fields)
	err := c.CreateCollection(ctx, schema, ut.DefaultShards)
	t.Log(err)
}
