package main

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func main() {
	fmt.Println("using default database...")
	ctx := context.Background()
	cfg := client.Config{
		Address:  "localhost:19530",
		Username: "root",
		Password: "S6.Zr9:Xk,/g7ByR,eb*XO<$KS~D7k|M",
	}
	clientDefault := mustConnect(ctx, cfg)
	defer clientDefault.Close()
	createCollection(ctx, clientDefault, "col1")
	if err := clientDefault.CreateDatabase(ctx, "db1"); err != nil {
		log.Fatalf("create db1 failed, %+v", err)
	}
	dbs, err := clientDefault.ListDatabases(ctx)
	if err != nil {
		log.Fatalf("list database failed: %+v", err)
	}

	fmt.Println("using db1...")
	cfg.DBName = ""
	cfg.Address = "localhost:19530/db1"
	clientDB1 := mustConnect(ctx, cfg)
	defer clientDB1.Close()
	createCollection(ctx, clientDB1, "col1")
	createCollection(ctx, clientDB1, "col2")

	fmt.Println("create collection col1 in default database...")
	collections, err := clientDefault.ListCollections(ctx)
	if err != nil {
		log.Fatalf("list on default database failed: %+v", err)
	}
	if len(collections) != 1 {
		log.Fatalf("unexpected count of default database, %d", len(collections))
	}
	if collections[0].Name != "col1" {
		log.Fatalf("unexpected db name of default database, %s", collections[0].Name)
	}

	fmt.Println("create collection col1, col2 in db1...")
	collections, err = clientDB1.ListCollections(ctx)
	if err != nil {
		log.Fatalf("list on db1 failed: %+v", err)
	}

	fmt.Println("create db2...")
	if err := clientDefault.CreateDatabase(ctx, "db2"); err != nil {
		log.Fatalf("create db2 failed, %+v", err)
	}
	dbs, err = clientDefault.ListDatabases(ctx)
	if err != nil {
		log.Fatalf("list database failed: %+v", err)
	}

	fmt.Println("connect to db2 with existing client...")
	clientDefault.UsingDatabase(ctx, "db2")
	fmt.Println("create and drop collection on db2...")
	createCollection(ctx, clientDefault, "col1")
	if err := clientDefault.DropCollection(ctx, "col1"); err != nil {
		log.Fatalf("drop col1 at db2 failed, %+v", err)
	}

	err = clientDefault.UsingDatabase(ctx, "")
	if err != nil {
		log.Fatalf("using database failed: %+v", err)
	}

	fmt.Println("drop db2: drop empty database should be always success...")
	if err := clientDefault.DropDatabase(ctx, "db2"); err != nil {
		log.Fatalf("drop db2 failed, %+v", err)
	}
	dbs, err = clientDefault.ListDatabases(ctx)
	if err != nil {
		log.Fatalf("list database failed: %+v", err)
	}

	fmt.Println("drop db1: drop non-empty database should be fail...")
	if err := clientDB1.DropDatabase(ctx, "db1"); err == nil {
		log.Fatalf("drop an non empty db success")
	}

	fmt.Println("drop db1: drop all collection before drop db...")
	if err := clientDB1.DropCollection(ctx, "col1"); err != nil {
		log.Fatalf("drop col1 at db1 failed, %+v", err)
	}
	if err := clientDB1.DropCollection(ctx, "col2"); err != nil {
		log.Fatalf("drop col2 at db1 failed, %+v", err)
	}
	if err := clientDB1.DropDatabase(ctx, "db1"); err != nil {
		log.Fatalf("drop db1 failed, %+v", err)
	}

	fmt.Println("Drop collection on default database...")
	if err := clientDefault.DropCollection(ctx, "col1"); err != nil {
		log.Fatalf("drop col1 at default db failed, %+v", err)
	}

	fmt.Println("drop default: drop default database should be fail...")
	if err := clientDefault.DropDatabase(ctx, ""); err == nil {
		log.Fatalf("drop an default db success")
	}
	dbs, err = clientDefault.ListDatabases(ctx)
	if err != nil {
		log.Fatalf("list database failed: %+v", err)
	}

	fmt.Println("db:", dbs)
}

func mustConnect(ctx context.Context, cfg client.Config) client.Client {
	c, err := client.NewClient(ctx, cfg)
	if err != nil {
		log.Fatalf("connect to database failed, %+v", err)
	}
	return c
}

func createCollection(ctx context.Context, c client.Client, collectionName string) {
	ok, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("%v", err)
	}
	if ok {
		c.DropCollection(ctx, collectionName)
	}
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "database demo",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "data",
				DataType:   entity.FieldTypeDouble,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     "embeddings",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: fmt.Sprintf("%d", 128),
				},
			},
		},
	}
	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		log.Fatalf("create collection failed, err: %v", err)
	}
}
