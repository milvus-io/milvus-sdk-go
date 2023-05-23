package testcases

import (
	"context"
	"flag"
	"log"
	"math/rand"
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
func createDefaultCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultFields(autoID)
	schema := common.GenSchema(collName, autoID, fields)

	// create default collection with fields: [int64, float, floatVector] and vector dim is default 128
	errCreateCollection := mc.CreateCollection(ctx, schema, common.DefaultShards)
	common.CheckErr(t, errCreateCollection, true)

	// close connect and drop collection after each case
	t.Cleanup(func() {
		mc.DropCollection(ctx, collName)
	})
	return collName
}

// create default collection
func createDefaultBinaryCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, dim string) string {
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
func createDefaultVarcharCollection(ctx context.Context, t *testing.T, mc *base.MilvusClient) string {
	t.Helper()

	// prepare schema
	collName := common.GenRandomString(6)
	fields := common.GenDefaultVarcharFields(false)
	schema := common.GenSchema(collName, false, fields)

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

func createCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, autoID bool, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultCollection(ctx, t, mc, autoID)

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
	collName := createDefaultBinaryCollection(ctx, t, mc, autoID, common.DefaultDimStr)

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

func createVarcharCollectionWithDataIndex(ctx context.Context, t *testing.T, mc *base.MilvusClient, withIndex bool) (string, entity.Column) {
	// collection
	collName := createDefaultVarcharCollection(ctx, t, mc)

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

	// prepare data
	int64Values := make([]int64, 0, nb)
	boolValues := make([]bool, 0, nb)
	int8Values := make([]int8, 0, nb)
	int16Values := make([]int16, 0, nb)
	int32Values := make([]int32, 0, nb)
	floatValues := make([]float32, 0, nb)
	doubleValues := make([]float64, 0, nb)
	varcharValues := make([]string, 0, nb)
	floatVectors := make([][]float32, 0, nb)
	for i := start; i < nb+start; i++ {
		int64Values = append(int64Values, int64(i))
		boolValues = append(boolValues, i/2 == 0)
		int8Values = append(int8Values, int8(i))
		int16Values = append(int16Values, int16(i))
		int32Values = append(int32Values, int32(i))
		floatValues = append(floatValues, float32(i))
		doubleValues = append(doubleValues, float64(i))
		varcharValues = append(varcharValues, strconv.Itoa(i))
		vec := make([]float32, 0, common.DefaultDim)
		for j := 0; j < common.DefaultDim; j++ {
			vec = append(vec, rand.Float32())
		}
		floatVectors = append(floatVectors, vec)
	}

	// insert data
	ids, errInsert := mc.Insert(
		ctx,
		collName,
		"",
		entity.NewColumnInt64("int64", int64Values),
		entity.NewColumnBool("bool", boolValues),
		entity.NewColumnInt8("int8", int8Values),
		entity.NewColumnInt16("int16", int16Values),
		entity.NewColumnInt32("int32", int32Values),
		entity.NewColumnFloat("float", floatValues),
		entity.NewColumnDouble("double", doubleValues),
		entity.NewColumnVarChar("varchar", varcharValues),
		entity.NewColumnFloatVector("floatVec", common.DefaultDim, floatVectors),
	)
	common.CheckErr(t, errInsert, true)
	require.Equal(t, nb, ids.Len())
	return collName, ids
}

func createInsertTwoPartitions(ctx context.Context, t *testing.T, mc *base.MilvusClient, collName string, nb int) (string, entity.Column, entity.Column) {
	// create new partition
	partitionName := "new"
	mc.CreatePartition(ctx, collName, partitionName)

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

	return partitionName, idsDefault, idsPartition
}

func TestMain(m *testing.M) {
	flag.Parse()
	log.Printf("parse addr=%s", *addr)
	code := m.Run()
	teardown()
	os.Exit(code)
}
