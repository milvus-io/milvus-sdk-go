package main

import (
	"context"
	"log"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr     = `localhost:19530`
	nEntities, dim = 10000, 128
	collectionName = "hello_partition_key"

	idCol, keyCol, embeddingCol = "ID", "key", "embeddings"
	topK                        = 3
)

func main() {
	ctx := context.Background()

	log.Println("start connecting to Milvus")
	c, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatalf("failed to connect to milvus, err: %v", err)
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
	log.Printf("create collection `%s`\n", collectionName)
	schema := entity.NewSchema().WithName(collectionName).WithDescription("hello_partition_key is the a demo to introduce the partition key related APIs").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName(keyCol).WithDataType(entity.FieldTypeInt64).WithIsPartitionKey(true)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber, client.WithPartitionNum(32)); err != nil { // use default shard number
		log.Fatalf("create collection failed, err: %v", err)
	}

	var keyList []int64
	var embeddingList [][]float32
	keyList = make([]int64, 0, nEntities)
	embeddingList = make([][]float32, 0, nEntities)
	for i := 0; i < nEntities; i++ {
		keyList = append(keyList, rand.Int63()%512)
	}
	for i := 0; i < nEntities; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}
		embeddingList = append(embeddingList, vec)
	}
	keyColData := entity.NewColumnInt64(keyCol, keyList)
	embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)

	log.Println("start to insert data into collection")

	if _, err := c.Insert(ctx, collectionName, "", keyColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `%s`, err: %v", collectionName, err)
	}

	log.Println("insert data done, start to flush")

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}
	log.Println("flush data done")

	// build index
	log.Println("start creating index HNSW")
	idx, err := entity.NewIndexHNSW(entity.L2, 16, 256)
	if err != nil {
		log.Fatalf("failed to create ivf flat index, err: %v", err)
	}
	if err := c.CreateIndex(ctx, collectionName, embeddingCol, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	log.Printf("build HNSW index done for collection `%s`\n", collectionName)
	log.Printf("start to load collection `%s`\n", collectionName)

	// load collection
	if err := c.LoadCollection(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	log.Println("load collection done")

	vec2search := []entity.Vector{
		entity.FloatVector(embeddingList[len(embeddingList)-2]),
		entity.FloatVector(embeddingList[len(embeddingList)-1]),
	}
	begin := time.Now()
	sp, _ := entity.NewIndexHNSWSearchParam(30)
	_, err = c.Search(ctx, collectionName, nil, "", []string{keyCol}, vec2search,
		embeddingCol, entity.L2, topK, sp)
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	log.Printf("search `%s` done, latency %v\n", collectionName, time.Since(begin))

	c.DropCollection(ctx, collectionName)
}
