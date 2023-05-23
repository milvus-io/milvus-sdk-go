package testcases

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestDatabase(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	clientDefault := createMilvusClient(ctx, t)
	defer clientDefault.Close()

	err := clientDefault.CreateDatabase(ctx, "db1")
	common.CheckErr(t, err, true)

	dbs, err := clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	if len(dbs) != 2 {
		t.Fatalf("unexpected db number, %d, %+v", len(dbs), dbs)
	}
	clientDB1 := createMilvusClient(ctx, t, client.Config{
		DBName: "db1",
	})
	defer clientDB1.Close()
	db1Col1, _ := createCollectionAllFields(ctx, t, clientDB1, common.DefaultNb, 0)
	db1Col2, _ := createCollectionAllFields(ctx, t, clientDB1, common.DefaultNb, 0)
	collections, err := clientDB1.ListCollections(ctx)
	common.CheckErr(t, err, true)
	if len(collections) != 2 {
		t.Fatalf("unexpected count of db1, %d", len(collections))
	}

	clientDefault.CreateDatabase(ctx, "db2")
	common.CheckErr(t, err, true)
	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	if len(dbs) != 3 {
		t.Fatalf("unexpected db number, %d, %+v", len(dbs), dbs)
	}

	clientDefault.UsingDatabase(ctx, "db2")
	db2Col1, _ := createCollectionAllFields(ctx, t, clientDefault, common.DefaultNb, 0)
	err = clientDefault.DropCollection(ctx, db2Col1)
	common.CheckErr(t, err, true)

	clientDefault.UsingDatabase(ctx, "")
	err = clientDefault.DropDatabase(ctx, "db2")
	common.CheckErr(t, err, true)

	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	if len(dbs) != 2 {
		t.Fatalf("unexpected db number, %d, %+v", len(dbs), dbs)
	}

	err = clientDB1.DropDatabase(ctx, "db1")
	common.CheckErr(t, err, false, "must drop all collections before drop database")

	err = clientDB1.DropCollection(ctx, db1Col1)
	common.CheckErr(t, err, true)

	err = clientDB1.DropCollection(ctx, db1Col2)
	common.CheckErr(t, err, true)

	err = clientDB1.DropDatabase(ctx, "db1")
	common.CheckErr(t, err, true)

	err = clientDefault.DropDatabase(ctx, "default")
	common.CheckErr(t, err, false, "can not drop default database")

	dbs, err = clientDefault.ListDatabases(ctx)
	common.CheckErr(t, err, true)
	if len(dbs) != 1 {
		t.Fatalf("unexpected db number, %d, %+v", len(dbs), dbs)
	}
}
