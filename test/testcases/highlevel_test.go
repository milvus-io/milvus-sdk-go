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
	// TODO why the index name is _default_idx_101 default is _default_idx_102 ?
	expIndex := entity.NewGenericIndex("_default_idx_101", entity.AUTOINDEX, DefaultVectorFieldName, expParams)
	common.CheckIndexResult(t, indexes, expIndex)

	// check collection is loaded
	loadState, _ := mc.GetLoadState(ctx, collName, []string{})
	require.Equal(t, entity.LoadStateLoaded, loadState)

	// insert
	pkColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeInt64, DefaultPkFieldName)
	vecColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeFloatVector, DefaultVectorFieldName, common.WithVectorDim(common.DefaultDim))
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
		entity.NewColumnInt64(DefaultPkFieldName, pkColumn.(*entity.ColumnInt64).Data()[:10]),
	)
	common.CheckErr(t, err, true)
	common.CheckOutputFields(t, queryResult, []string{DefaultPkFieldName, DefaultVectorFieldName})
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnInt64(DefaultPkFieldName, pkColumn.(*entity.ColumnInt64).Data()[:10]),
		entity.NewColumnFloatVector(DefaultVectorFieldName, int(common.DefaultDim), vecColumn.(*entity.ColumnFloatVector).Data()[:10]),
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
	expIndex := entity.NewGenericIndex("_default_idx_101", entity.AUTOINDEX, DefaultVectorFieldName, expParams)
	common.CheckIndexResult(t, indexes, expIndex)

	// check collection is loaded
	loadState, _ := mc.GetLoadState(ctx, collName, []string{})
	require.Equal(t, entity.LoadStateLoaded, loadState)

	// insert
	pkColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeVarChar, pkFieldName)
	vecColumn := common.GenColumnData(0, common.DefaultNb, entity.FieldTypeFloatVector, vectorFieldName, common.WithVectorDim(common.DefaultDim))
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
		entity.NewColumnVarChar(pkFieldName, pkColumn.(*entity.ColumnVarChar).Data()[:10]),
	)
	common.CheckErr(t, err, true)
	common.CheckOutputFields(t, queryResult, []string{pkFieldName, vectorFieldName})
	common.CheckQueryResult(t, queryResult, []entity.Column{
		entity.NewColumnVarChar(pkFieldName, pkColumn.(*entity.ColumnVarChar).Data()[:10]),
		entity.NewColumnFloatVector(vectorFieldName, int(common.DefaultDim), vecColumn.(*entity.ColumnFloatVector).Data()[:10]),
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
