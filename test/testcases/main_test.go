package testcases

import (
	"context"
	"flag"
	"log"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

var addr = flag.String("addr", "localhost:19530", "server host and port")

func init() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
}

// teardown
func teardown() {
	log.Println("Start to tear down all")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	mc, err := base.NewDefaultMilvusClient(ctx, *addr)
	if err != nil {
		log.Fatalf("teardown failed to connect milvus with error %v", err)
	}
	defer mc.Close()

	// clear dbs
	dbs, _ := mc.ListDatabases(ctx)
	for _, db := range dbs {
		if db.Name != common.DefaultDb {
			_ = mc.UsingDatabase(ctx, db.Name)
			collections, _ := mc.ListCollections(ctx)
			for _, coll := range collections {
				_ = mc.DropCollection(ctx, coll.Name)
			}
			_ = mc.DropDatabase(ctx, db.Name)
		}
	}
}

func createContext(t *testing.T, timeout time.Duration) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	t.Cleanup(func() {
		cancel()
	})
	return ctx
}

// create connect
func createMilvusClient(ctx context.Context, t *testing.T, cfg ...client.Config) *base.MilvusClient {
	t.Helper()

	var (
		mc  *base.MilvusClient
		err error
	)
	if len(cfg) == 0 {
		mc, err = base.NewDefaultMilvusClient(ctx, *addr)
	} else {
		cfg[0].Address = *addr
		mc, err = base.NewMilvusClient(ctx, cfg[0])
	}
	common.CheckErr(t, err, true)

	t.Cleanup(func() {
		mc.Close()
	})

	return mc
}

// create default collection
func createCustomerCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, schema *entity.Schema,
	shardsNum int32, opts ...client.CreateCollectionOption) {
	t.Helper()

	// create default collection with customer schema
	errCreateCollection := mc.CreateCollection(ctx, schema, shardsNum, opts...)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		_ = mc.DropCollection(ctx, schema.CollectionName)
	})
}

// create default collection
func createDefaultCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, shards int32, opts ...client.CreateCollectionOption) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultFields(autoID)
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, shards, opts...)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
	})
	return collName
}

// create default collection
func createDefaultBinaryCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, dim int64) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultBinaryFields(autoID, dim)
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
		// mc.Close()
	})
	return collName
}

// create default varchar pk collection
func createDefaultVarcharCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, opts ...client.CreateCollectionOption) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultVarcharFields(false)
	schema := common.GenSchema(collName, false, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards, opts...)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
		// mc.Close()
	})
	return collName
}

func createCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool, opts ...client.CreateCollectionOption) (string, entity.Column) {
	// collection
	collName := createDefaultCollection(ctx, t, mc, autoID, common.DefaultShards, opts...)
	// insert data
	var ids entity.Column
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, common.DefaultNb, common.DefaultDim)
	if autoID {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		ids = pk
	} else {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, intColumn, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		common.CheckInsertResult(t, pk, intColumn)
		ids = pk
	}

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexHNSW(entity.L2, 8, 96)
		err := mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false, client.WithIndexName(""))
		common.CheckErr(t, err, true)
	}
	return collName, ids
}

func createBinaryCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultBinaryCollection(ctx, t, mc, autoID, common.DefaultDim)

	// insert data
	var ids entity.Column
	intColumn, floatColumn, vecColumn := common.GenDefaultBinaryData(0, common.DefaultNb, common.DefaultDim)
	if autoID {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		ids = pk
	} else {
		pk, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, intColumn, floatColumn, vecColumn)
		common.CheckErr(t, errInsert, true)
		common.CheckInsertResult(t, pk, intColumn)
		ids = pk
	}

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName(""))
		common.CheckErr(t, err, true)
	}
	return collName, ids
}

func createVarcharCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, withIndex bool, opts ...client.CreateCollectionOption) (string, entity.Column) {
	// collection
	collName := createDefaultVarcharCollection(ctx, t, mc, opts...)

	// insert data
	varcharColumn, vecColumn := common.GenDefaultVarcharData(0, common.DefaultNb, common.DefaultDim)
	ids, errInsert := mc.Insert(ctx, collName, common.DefaultPartition, varcharColumn, vecColumn)
	common.CheckErr(t, errInsert, true)
	common.CheckInsertResult(t, ids, varcharColumn)

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)

	// create index
	if withIndex {
		idx, _ := entity.NewIndexBinIvfFlat(entity.JACCARD, 128)
		err := mc.CreateIndex(ctx, collName, common.DefaultBinaryVecFieldName, idx, false, client.WithIndexName(""))
		common.CheckErr(t, err, true)
	}
	return collName, ids
}

const (
	Int64FloatVec      CollectionFieldsType = "PkInt64FloatVec"     // int64 + float + floatVec
	Int64BinaryVec     CollectionFieldsType = "Int64BinaryVec"      // int64 + float + binaryVec
	VarcharBinaryVec   CollectionFieldsType = "PkVarcharBinaryVec"  // varchar + binaryVec
	Int64FloatVecJSON  CollectionFieldsType = "PkInt64FloatVecJson" // int64 + float + floatVec + json
	Int64FloatVecArray CollectionFieldsType = "Int64FloatVecArray"  // int64 + float + floatVec + all array
	AllVectors         CollectionFieldsType = "AllVectors"          // int64 + fp32Vec + fp16Vec + binaryVec
	AllFields          CollectionFieldsType = "AllFields"           // all scalar fields + floatVec
)

func createCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, cp CollectionParams, opts ...client.CreateCollectionOption) string {
	collName := common.GenRandomString(4)
	var fields []*entity.Field
	// fields
	switch cp.CollectionFieldsType {
	// int64 + float + floatVec
	case Int64FloatVec:
		fields = common.GenDefaultFields(cp.AutoID)
		// int64 + float + binaryVec
	case Int64BinaryVec:
		fields = common.GenDefaultBinaryFields(cp.AutoID, cp.Dim)
	case VarcharBinaryVec:
		fields = common.GenDefaultVarcharFields(cp.AutoID)
	case Int64FloatVecJSON:
		fields = common.GenDefaultFields(cp.AutoID)
		jsonField := common.GenField(common.DefaultJSONFieldName, entity.FieldTypeJSON)
		fields = append(fields, jsonField)
	case Int64FloatVecArray:
		fields = common.GenDefaultFields(cp.AutoID)
		fields = append(fields, common.GenAllArrayFieldsWithCapacity(cp.MaxCapacity)...)
	case AllVectors:
		fields = []*entity.Field{
			common.GenField(common.DefaultIntFieldName, entity.FieldTypeInt64, common.WithIsPrimaryKey(true), common.WithAutoID(cp.AutoID)),
			common.GenField(common.DefaultFloatVecFieldName, entity.FieldTypeFloatVector, common.WithDim(cp.Dim)),
			common.GenField(common.DefaultFloat16VecFieldName, entity.FieldTypeFloat16Vector, common.WithDim(cp.Dim)),
			common.GenField(common.DefaultBFloat16VecFieldName, entity.FieldTypeBFloat16Vector, common.WithDim(cp.Dim)),
			common.GenField(common.DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, common.WithDim(cp.Dim)),
		}
	case AllFields:
		fields = common.GenAllFields()
	}

	// schema
	schema := common.GenSchema(collName, cp.AutoID, fields, common.WithEnableDynamicField(cp.EnableDynamicField))

	// create collection
	err := mc.CreateCollection(ctx, schema, cp.ShardsNum, opts...)
	common.CheckErr(t, err, true)

	t.Cleanup(func() {
		_ = mc.DropCollection(ctx, collName)
	})

	return collName
}

func insertData(ctx context.Context, t *testing.T, mc *base.MilvusClient, dp DataParams, opts ...common.GenColumnDataOption) (entity.Column, error) {
	// todo autoid
	// prepare data
	var data []entity.Column
	rows := make([]interface{}, 0, dp.nb)
	switch dp.CollectionFieldsType {

	// int64 + float + floatVec
	case Int64FloatVec:
		if dp.WithRows {
			rows = common.GenDefaultRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		} else {
			intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(dp.start, dp.nb, dp.dim)
			data = append(data, intColumn, floatColumn, vecColumn)
		}

		// int64 + float + binaryVec
	case Int64BinaryVec:
		if dp.WithRows {
			rows = common.GenDefaultBinaryRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		} else {
			intColumn, floatColumn, binaryColumn := common.GenDefaultBinaryData(dp.start, dp.nb, dp.dim)
			data = append(data, intColumn, floatColumn, binaryColumn)
		}
	// varchar + binary
	case VarcharBinaryVec:
		if dp.WithRows {
			rows = common.GenDefaultVarcharRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		} else {
			varcharColumn, binaryColumn := common.GenDefaultVarcharData(dp.start, dp.nb, dp.dim)
			data = append(data, varcharColumn, binaryColumn)
		}

		// default + json
	case Int64FloatVecJSON:
		if dp.WithRows {
			rows = common.GenDefaultJSONRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		} else {
			intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(dp.start, dp.nb, dp.dim)
			jsonColumn := common.GenDefaultJSONData(common.DefaultJSONFieldName, dp.start, dp.nb)
			data = append(data, intColumn, floatColumn, vecColumn, jsonColumn)
		}
	case Int64FloatVecArray:
		if dp.WithRows {
			rows = common.GenDefaultArrayRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		} else {
			data = common.GenAllArrayData(dp.start, dp.nb, opts...)
			intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(dp.start, dp.nb, dp.dim)
			data = append(data, intColumn, floatColumn, vecColumn)
		}

	case AllVectors:
		if dp.WithRows {
			rows = common.GenAllVectorsRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField)
		}
		data = common.GenAllVectorsData(dp.start, dp.nb, dp.dim, opts...)
	case AllFields:
		if dp.WithRows {
			rows = common.GenAllFieldsRows(dp.start, dp.nb, dp.dim, dp.EnableDynamicField, opts...)
		}
		data = common.GenAllFieldsData(dp.start, dp.nb, dp.dim, opts...)
	}

	if dp.EnableDynamicField && !dp.WithRows {
		data = append(data, common.GenDynamicFieldData(dp.start, dp.nb)...)
	}

	// insert
	var ids entity.Column
	var err error
	if dp.WithRows {
		ids, err = mc.InsertRows(ctx, dp.CollectionName, dp.PartitionName, rows)
	} else {
		ids, err = mc.Insert(ctx, dp.CollectionName, dp.PartitionName, data...)
	}
	common.CheckErr(t, err, true)
	require.Equalf(t, dp.nb, ids.Len(), "Expected insert id num: %d, actual: ", dp.nb, ids.Len())
	return ids, err
}

// create collection with all scala fields and insert data without flush
func createCollectionAllFields(ctx context.Context, t *testing.T, mc *base.MilvusClient, nb int, start int) (string, entity.Column) {
	t.Helper()

	// prepare fields, name, schema
	allFields := common.GenAllFields()
	collName := common.GenRandomString(6)
	schema := common.GenSchema(collName, false, allFields)

	// create collection
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	data := common.GenAllFieldsData(start, nb, common.DefaultDim, common.WithArrayCapacity(common.TestCapacity))
	ids, errInsert := mc.Insert(ctx, collName, "", data...)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())
	return collName, ids
}

func createInsertTwoPartitions(ctx context.Context, t *testing.T, mc *base.MilvusClient, collName string, nb int) (partitionName string, defaultPartition HelpPartitionColumns, newPartition HelpPartitionColumns) {
	// create new partition
	partitionName = "new"
	_ = mc.CreatePartition(ctx, collName, partitionName)

	// insert nb into default partition, pks from 0 to nb
	intColumn, floatColumn, vecColumn := common.GenDefaultColumnData(0, nb, common.DefaultDim)
	idsDefault, _ := mc.Insert(ctx, collName, common.DefaultPartition, intColumn, floatColumn, vecColumn)

	// insert nb into new partition, pks from nb to nb*2
	intColumnNew, floatColumnNew, vecColumnNew := common.GenDefaultColumnData(nb, nb, common.DefaultDim)
	idsPartition, _ := mc.Insert(ctx, collName, partitionName, intColumnNew, floatColumnNew, vecColumnNew)

	// flush
	errFlush := mc.Flush(ctx, collName, false)
	common.CheckErr(t, errFlush, true)
	stats, _ := mc.GetCollectionStatistics(ctx, collName)
	require.Equal(t, strconv.Itoa(nb*2), stats[common.RowCount])

	defaultPartition = HelpPartitionColumns{
		PartitionName: common.DefaultPartition,
		IdsColumn:     idsDefault,
		VectorColumn:  vecColumn,
	}

	newPartition = HelpPartitionColumns{
		PartitionName: partitionName,
		IdsColumn:     idsPartition,
		VectorColumn:  vecColumnNew,
	}

	return partitionName, defaultPartition, newPartition
}

// prepare collection, maybe data index and load
func prepareCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, collParam CollectionParams, opts ...PrepareCollectionOption) string {
	// default insert nb entities with 0 start
	defaultDp := DataParams{DoInsert: true, CollectionName: "", PartitionName: "", CollectionFieldsType: collParam.CollectionFieldsType,
		start: 0, nb: common.DefaultNb, dim: collParam.Dim, EnableDynamicField: collParam.EnableDynamicField, WithRows: false}

	// default do flush
	defaultFp := FlushParams{DoFlush: true, PartitionNames: []string{}, async: false}

	// default build index
	idx, err := entity.NewIndexHNSW(entity.L2, 8, 96)
	common.CheckErr(t, err, true)
	defaultIndexParams := []IndexParams{{BuildIndex: true, Index: idx, FieldName: common.DefaultFloatVecFieldName, async: false}}

	// default load collection
	defaultLp := LoadParams{DoLoad: true, async: false}
	opt := &ClientParamsOption{
		DataParams:  defaultDp,
		FlushParams: defaultFp,
		IndexParams: defaultIndexParams,
		LoadParams:  defaultLp,
	}
	for _, o := range opts {
		o(opt)
	}
	// create collection
	collName := createCollection(ctx, t, mc, collParam, opt.CreateOpts)

	// insert
	if opt.DataParams.DoInsert {
		if opt.DataParams.EnableDynamicField != collParam.EnableDynamicField {
			t.Fatalf("The EnableDynamicField of CollectionParams and DataParams should be equal.")
		}
		opt.DataParams.CollectionName = collName
		opt.DataParams.CollectionFieldsType = collParam.CollectionFieldsType
		insertData(ctx, t, mc, opt.DataParams)
	}

	// flush
	if opt.FlushParams.DoFlush {
		err := mc.Flush(ctx, collName, opt.FlushParams.async)
		common.CheckErr(t, err, true)
	}

	// index
	for _, idxParams := range opt.IndexParams {
		if idxParams.BuildIndex {
			var err error
			if opt.IndexOpts == nil {
				err = mc.CreateIndex(ctx, collName, idxParams.FieldName, idxParams.Index, idxParams.async)
			} else {
				err = mc.CreateIndex(ctx, collName, idxParams.FieldName, idxParams.Index, idxParams.async, opt.IndexOpts)
			}
			common.CheckErr(t, err, true)
		}
	}

	// load
	if opt.LoadParams.DoLoad {
		var err error
		if len(opt.LoadParams.PartitionNames) > 0 {
			err = mc.LoadPartitions(ctx, collName, opt.LoadParams.PartitionNames, opt.LoadParams.async)
			common.CheckErr(t, err, true)
		} else {
			if opt.LoadOpts != nil {
				err = mc.LoadCollection(ctx, collName, opt.LoadParams.async, opt.LoadOpts)
			} else {
				err = mc.LoadCollection(ctx, collName, opt.LoadParams.async)
			}
			common.CheckErr(t, err, true)
		}
	}
	return collName
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.Printf("parse addr=%s", *addr)
	code := m.Run()
	teardown()
	os.Exit(code)
}
