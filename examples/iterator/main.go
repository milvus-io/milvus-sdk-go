package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr     = `localhost:19530`
	nEntities, dim = 3000, 128
	collectionName = "hello_iterator"

	msgFmt                         = "==== %s ====\n"
	idCol, randomCol, embeddingCol = "ID", "random", "embeddings"
	topK                           = 3
)

func main() {
	ctx := context.Background()

	log.Printf(msgFmt, "start connecting to Milvus")
	c, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus, err: ", err.Error())
	}
	defer c.Close()

	// delete collection if exists
	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("failed to check collection exists, err: %v", err)
	}
	if has {
		c.DropCollection(ctx, collectionName)
	}

	// create collection
	log.Printf(msgFmt, fmt.Sprintf("create collection, `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription("hello_milvus is the simplest demo to introduce the APIs").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName(randomCol).WithDataType(entity.FieldTypeDouble)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil { // use default shard number
		log.Fatalf("create collection failed, err: %v", err)
	}

	// build index
	log.Printf(msgFmt, "start creating index IVF_FLAT")
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		log.Fatalf("failed to create ivf flat index, err: %v", err)
	}
	if err := c.CreateIndex(ctx, collectionName, embeddingCol, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	log.Printf(msgFmt, "start loading collection")
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	// insert data
	log.Printf(msgFmt, "start inserting random entities")
	idList, randomList := make([]int64, 0, nEntities), make([]float64, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	// generate data
	for i := 0; i < nEntities; i++ {
		idList = append(idList, int64(i))
	}
	for i := 0; i < nEntities; i++ {
		randomList = append(randomList, rand.Float64())
	}
	for i := 0; i < nEntities; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}
		embeddingList = append(embeddingList, vec)
	}
	idColData := entity.NewColumnInt64(idCol, idList)
	randomColData := entity.NewColumnDouble(randomCol, randomList)
	embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)

	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `hello_milvus, err: %v", err)
	}

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}

	itr, err := c.QueryIterator(ctx, client.NewQueryIteratorOption(collectionName).WithOutputFields(idCol).WithBatchSize(100))
	if err != nil {
		log.Fatal("failed to query iterator: ", err.Error())
	}
	for {
		rs, err := itr.Next(ctx)
		if err != nil {
			if err == io.EOF {
				log.Println("iterator reach EOF")
				break
			}
			log.Fatal("failed to query iterator. next: ", err.Error())
		}
		var idlist []int64
		for _, col := range rs {
			if col.Name() == idCol {
				idColumn := col.(*entity.ColumnInt64)
				for i := 0; i < col.Len(); i++ {
					val, err := idColumn.ValueByIdx(i)
					if err != nil {
						log.Fatal(err)
					}
					idlist = append(idlist, val)
				}
			}
		}
		log.Printf("\tids: %#v\n", idlist)
	}

	// drop collection
	log.Printf(msgFmt, "drop collection `hello_milvus`")
	if err := c.DropCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to drop collection, err: %v", err)
	}
}
