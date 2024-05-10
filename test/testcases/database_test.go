//go:build L0

package testcases

import (
	"fmt"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// teardownTest
func teardownTest(t *testing.T) func(t *testing.T) {
	log.Println("setup test func")
	return func(t *testing.T) {
		log.Println("teardown func drop all non-default db")
		// drop all db
		ctx := createContext(t, time.Second*common.DefaultTimeout)
		mc := createMilvusClient(ctx, t)
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
}

func TestDatabase(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)

	ctx := createContext(t, time.Second*common.DefaultTimeout)
	clientDefault := createMilvusClient(ctx, t)
	tmpNb := 100
	// create db1
	dbName1 := common.GenRandomString(4)
	err := clientDefault.CreateDatabase(ctx, dbName1)
	common.CheckErr(t, err, true)

	// list db and verify db1 in dbs
	dbs, errList := clientDefault.ListDatabases(ctx)
	common.CheckErr(t, errList, true)
	common.CheckContainsDb(t, dbs, dbName1)

	// new client with db1 -> create collections
	clientDB1 := createMilvusClient(ctx, t, client.Config{
		DBName: dbName1,
	})
	db1Col1, _ := createCollectionAllFields(ctx, t, clientDB1, tmpNb, 0)
	db1Col2, _ := createCollectionAllFields(ctx, t, clientDB1, tmpNb, 0)
	collections, errListCollections := clientDB1.ListCollections(ctx)
	common.CheckErr(t, errListCollections, true)
	common.CheckContainsCollection(t, collections, db1Col1)
	common.CheckContainsCollection(t, collections, db1Col2)

	// create db2
	dbName2 := common.GenRandomString(4)
	err = clientDefault.CreateDatabase(ctx, dbName2)
	common.CheckErr(t, err, true)
	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	common.CheckContainsDb(t, dbs, dbName2)

	// using db2 -> create collection -> drop collection
	clientDefault.UsingDatabase(ctx, dbName2)
	db2Col1, _ := createCollectionAllFields(ctx, t, clientDefault, tmpNb, 0)
	err = clientDefault.DropCollection(ctx, db2Col1)
	common.CheckErr(t, err, true)

	// using empty db -> drop db2
	clientDefault.UsingDatabase(ctx, "")
	err = clientDefault.DropDatabase(ctx, dbName2)
	common.CheckErr(t, err, true)

	// list db and verify db drop success
	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	common.CheckNotContainsDb(t, dbs, dbName2)

	// drop db1 which has some collections
	err = clientDB1.DropDatabase(ctx, dbName1)
	common.CheckErr(t, err, false, "must drop all collections before drop database")

	// drop all db1's collections -> drop db1
	err = clientDB1.DropCollection(ctx, db1Col1)
	common.CheckErr(t, err, true)

	err = clientDB1.DropCollection(ctx, db1Col2)
	common.CheckErr(t, err, true)

	err = clientDB1.DropDatabase(ctx, dbName1)
	common.CheckErr(t, err, true)

	// drop default db
	err = clientDefault.DropDatabase(ctx, common.DefaultDb)
	common.CheckErr(t, err, false, "can not drop default database")

	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	common.CheckContainsDb(t, dbs, common.DefaultDb)
}

// test create with invalid db name
func TestCreateDb(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)

	// create db
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	dbName := common.GenRandomString(4)
	err := mc.CreateDatabase(ctx, dbName)
	common.CheckErr(t, err, true)

	// create existed db
	err = mc.CreateDatabase(ctx, dbName)
	common.CheckErr(t, err, false, fmt.Sprintf("database already exist: %s", dbName))

	// create default db
	err = mc.CreateDatabase(ctx, common.DefaultDb)
	common.CheckErr(t, err, false, fmt.Sprintf("database already exist: %s", common.DefaultDb))

	emptyErr := mc.CreateDatabase(ctx, "")
	common.CheckErr(t, emptyErr, false, "database name couldn't be empty")
}

// test drop db
func TestDropDb(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)

	// create collection in default db
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)
	collections, _ := mc.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collName)

	// create db
	dbName := common.GenRandomString(4)
	err := mc.CreateDatabase(ctx, dbName)
	common.CheckErr(t, err, true)

	// using db and drop the db
	err = mc.UsingDatabase(ctx, dbName)
	common.CheckErr(t, err, true)
	err = mc.DropDatabase(ctx, dbName)
	common.CheckErr(t, err, true)

	// verify current db
	_, err = mc.ListCollections(ctx)
	common.CheckErr(t, err, false, fmt.Sprintf("database not found[database=%s]", dbName))

	// using default db and verify collections
	mc.UsingDatabase(ctx, common.DefaultDb)
	collections, _ = mc.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collName)

	// drop not existed db
	err = mc.DropDatabase(ctx, common.GenRandomString(4))
	common.CheckErr(t, err, true)

	// drop empty db
	err = mc.DropDatabase(ctx, "")
	common.CheckErr(t, err, false, "database name couldn't be empty")

	// drop default db
	err = mc.DropDatabase(ctx, common.DefaultDb)
	common.CheckErr(t, err, false, "can not drop default database")
}

// test using db
func TestUsingDb(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)

	// create collection in default db
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	collName := createDefaultCollection(ctx, t, mc, true, common.DefaultShards)
	collections, _ := mc.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collName)

	// using not existed db
	dbName := common.GenRandomString(4)
	err := mc.UsingDatabase(ctx, dbName)
	common.CheckErr(t, err, false, fmt.Sprintf("connect fail, database not found[database=%s]", dbName))

	// using empty db
	err = mc.UsingDatabase(ctx, "")
	common.CheckErr(t, err, true)
	collections, _ = mc.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collName)

	// using current db
	err = mc.UsingDatabase(ctx, common.DefaultDb)
	common.CheckErr(t, err, true)
	collections, _ = mc.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collName)
}

// test client with db
func TestClientWithDb(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)
	ctx := createContext(t, time.Second*common.DefaultTimeout)

	// connect with not existed db
	_, err := base.NewMilvusClient(ctx, client.Config{
		Address: *addr,
		DBName:  common.GenRandomString(4),
	})
	common.CheckErr(t, err, false, "database not found")

	// connect default db -> create a collection in default db
	mcDefault, errDefault := base.NewMilvusClient(ctx, client.Config{
		Address: *addr,
		DBName:  common.DefaultDb,
	})
	common.CheckErr(t, errDefault, true)
	_, _ = mcDefault.ListCollections(ctx)
	collDefault := createDefaultCollection(ctx, t, mcDefault, true, common.DefaultShards)

	// create a db and create collection in db
	dbName := common.GenRandomString(5)
	err = mcDefault.CreateDatabase(ctx, dbName)
	common.CheckErr(t, err, true)

	// and connect with db
	mcDb, err := base.NewMilvusClient(ctx, client.Config{
		Address: *addr,
		DBName:  dbName,
	})
	common.CheckErr(t, err, true)
	collDb := createDefaultCollection(ctx, t, mcDb, true, common.DefaultShards)
	collections, _ := mcDb.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collDb)

	// using default db and collection not in
	_ = mcDb.UsingDatabase(ctx, common.DefaultDb)
	collections, _ = mcDb.ListCollections(ctx)
	common.CheckNotContainsCollection(t, collections, collDb)

	// connect empty db (actually default db)
	mcEmpty, err := base.NewMilvusClient(ctx, client.Config{
		Address: *addr,
		DBName:  "",
	})
	common.CheckErr(t, err, true)
	collections, _ = mcEmpty.ListCollections(ctx)
	common.CheckContainsCollection(t, collections, collDefault)
}

func TestAlterDatabase(t *testing.T) {
	teardownSuite := teardownTest(t)
	defer teardownSuite(t)

	// create db
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	dbName := common.GenRandomString(4)
	err := mc.CreateDatabase(ctx, dbName)
	common.CheckErr(t, err, true)

	dbInfo, err := mc.DescribeDatabase(ctx, dbName)
	common.CheckErr(t, err, true)
	assert.Equal(t, dbInfo.Name, dbName)

	err = mc.AlterDatabase(ctx, dbName, entity.DatabaseResourceGroups([]string{"rg1"}))
	common.CheckErr(t, err, true)

	dbInfo, err = mc.DescribeDatabase(ctx, dbName)
	common.CheckErr(t, err, true)
	assert.Equal(t, dbInfo.Name, dbName)
	assert.Len(t, dbInfo.Properties, 1)

	err = mc.AlterDatabase(ctx, dbName, entity.DatabaseReplica(1))
	common.CheckErr(t, err, true)

	dbInfo, err = mc.DescribeDatabase(ctx, dbName)
	common.CheckErr(t, err, true)
	assert.Equal(t, dbInfo.Name, dbName)
	assert.Len(t, dbInfo.Properties, 2)
}
