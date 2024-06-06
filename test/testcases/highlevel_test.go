//go:build L0

package testcases

import (
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

const (
	DefaultPkFieldName     = "id"
	DefaultVectorFieldName = "vector"
	nb                     = 10000
)

// test highlevel api new collection
func TestNewCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// new collection
	collName := common.GenRandomString(5)
	err := mc.NewCollection(ctx, collName, common.DefaultDim, client.WithConsistencyLevel(entity.ClStrong))
	common.CheckErr(t, err, true)

	// describe collection and check
	collection, _ := mc.DescribeCollection(ctx, collName)

	pkField := common.GenField(DefaultPkFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true))
	vecField := common.GenField(DefaultVectorFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim))
	expSchema := common.GenSchema(collName, false, []*entity.Field{pkField, vecField}, common.WithEnableDynamicField(true))
	common.CheckCollection(t, collection, collName, 1, expSchema, entity.ClStrong)

	// describe index and check
	indexes, _ := mc.DescribeIndex(ctx, collName, DefaultVectorFieldName)
	expParams := map[string]string{
		"metric_type": string(entity.IP),
		"index_type":  string(entity.AUTOINDEX),
	}
	expIndex := entity.NewGenericIndex(DefaultVectorFieldName, entity.AUTOINDEX, expParams)
	common.CheckIndexResult(t, indexes, expIndex)

	// check collection is loaded
	loadState, _ := mc.GetLoadState(ctx, collName, []string{})
	require.Equal(t, entity.LoadStateLoaded, loadState)

	// insert
	pkColumn := common.GenColumnData(0, nb, entity.FieldTypeInt64, DefaultPkFieldName)
	vecColumn := common.GenColumnData(0, nb, entity.FieldTypeFloatVector, DefaultVectorFieldName, common.WithVectorDim(common.DefaultDim))
	_, err = mc.Insert(
		ctx, collName, "",
		pkColumn, vecColumn,
	)
	common.CheckErr(t, err, true)
	//time.Sleep(10) // because consistence level

	// get
	queryResult, err := mc.Get(
		ctx,
		collName,
		pkColumn.Slice(0, 10),
	)
	common.CheckErr(t, err, true)
	common.CheckOutputFields(t, queryResult, []string{DefaultPkFieldName, DefaultVectorFieldName})
	common.CheckQueryResult(t, queryResult, []entity.Column{
		pkColumn.Slice(0, 10),
		vecColumn.Slice(0, 10),
	})

	// search
	sp, _ := entity.NewIndexAUTOINDEXSearchParam(2)
	searchRes, errSearchEmpty := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{DefaultPkFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		DefaultVectorFieldName,
		entity.IP,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchEmpty, true)
	common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
}

func TestNewCollectionCustomize(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)

	// new collection
	collName := common.GenRandomString(5)
	pkFieldName := "pk"
	vectorFieldName := "vec"
	err := mc.NewCollection(
		ctx, collName, common.DefaultDim,
		client.WithPKFieldName(pkFieldName), client.WithPKFieldType(entity.FieldTypeVarChar), client.WithPKMaxLength(2048),
		client.WithVectorFieldName(vectorFieldName),
		client.WithMetricsType(entity.L2), client.WithAutoID(false), client.WithEnableDynamicSchema(false),
		client.WithConsistencyLevel(entity.ClStrong))
	common.CheckErr(t, err, true)

	// describe collection and check
	collection, _ := mc.DescribeCollection(ctx, collName)

	pkField := common.GenField(pkFieldName, entity.FieldTypeVarChar, common.WithIsPrimaryKey(true), common.WithMaxLength(2048))
	vecField := common.GenField(vectorFieldName, entity.FieldTypeFloatVector, common.WithDim(common.DefaultDim))
	expSchema := common.GenSchema(collName, false, []*entity.Field{pkField, vecField})
	common.CheckCollection(t, collection, collName, 1, expSchema, entity.ClStrong)

	// describe index and check
	indexes, _ := mc.DescribeIndex(ctx, collName, vectorFieldName)
	expParams := map[string]string{
		"metric_type": string(entity.L2),
		"index_type":  string(entity.AUTOINDEX),
	}
	expIndex := entity.NewGenericIndex(vectorFieldName, entity.AUTOINDEX, expParams)
	common.CheckIndexResult(t, indexes, expIndex)

	// check collection is loaded
	loadState, _ := mc.GetLoadState(ctx, collName, []string{})
	require.Equal(t, entity.LoadStateLoaded, loadState)

	// insert
	pkColumn := common.GenColumnData(0, nb, entity.FieldTypeVarChar, pkFieldName)
	vecColumn := common.GenColumnData(0, nb, entity.FieldTypeFloatVector, vectorFieldName, common.WithVectorDim(common.DefaultDim))
	_, err = mc.Insert(
		ctx, collName, "",
		pkColumn,
		vecColumn,
	)
	common.CheckErr(t, err, true)

	// get
	queryResult, err := mc.Get(
		ctx,
		collName,
		pkColumn.Slice(0, 10),
	)
	common.CheckErr(t, err, true)
	common.CheckOutputFields(t, queryResult, []string{pkFieldName, vectorFieldName})
	common.CheckQueryResult(t, queryResult, []entity.Column{
		pkColumn.Slice(0, 10),
		vecColumn.Slice(0, 10),
	})

	// search
	sp, _ := entity.NewIndexAUTOINDEXSearchParam(2)
	searchRes, errSearchEmpty := mc.Search(
		ctx, collName,
		[]string{},
		"",
		[]string{pkFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		vectorFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	common.CheckErr(t, errSearchEmpty, true)
	common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)
}
