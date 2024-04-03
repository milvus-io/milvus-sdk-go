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
	collectionName = "hello_multi_vectors"

	idCol, keyCol, embeddingCol1, embeddingCol2 = "ID", "key", "vector1", "vector2"
	topK                                        = 3
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
		WithField(entity.NewField().WithName(keyCol).WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName(embeddingCol1).WithDataType(entity.FieldTypeFloatVector).WithDim(dim)).
		WithField(entity.NewField().WithName(embeddingCol2).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil { // use default shard number
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
	embeddingColData1 := entity.NewColumnFloatVector(embeddingCol1, dim, embeddingList)
	embeddingColData2 := entity.NewColumnFloatVector(embeddingCol2, dim, embeddingList)

	log.Println("start to insert data into collection")

	if _, err := c.Insert(ctx, collectionName, "", keyColData, embeddingColData1, embeddingColData2); err != nil {
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
	if err := c.CreateIndex(ctx, collectionName, embeddingCol1, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}
	if err := c.CreateIndex(ctx, collectionName, embeddingCol2, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	log.Printf("build HNSW index done for collection `%s`\n", collectionName)
	log.Printf("start to load collection `%s`\n", collectionName)

	// load collection
	if err := c.LoadCollection(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	log.Println("load collection done")

	// currently only nq =1 is supported
	vec2search1 := []entity.Vector{
		entity.FloatVector(embeddingList[len(embeddingList)-2]),
	}
	vec2search2 := []entity.Vector{
		entity.FloatVector(embeddingList[len(embeddingList)-1]),
	}

	begin := time.Now()
	sp, _ := entity.NewIndexHNSWSearchParam(30)

	log.Println("start to search vector field 1")
	result, err := c.Search(ctx, collectionName, nil, "", []string{keyCol, embeddingCol1, embeddingCol2}, vec2search1,
		embeddingCol1, entity.L2, topK, sp)
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	log.Printf("search `%s` done, latency %v\n", collectionName, time.Since(begin))
	for _, rs := range result {
		for i := 0; i < rs.ResultCount; i++ {
			id, _ := rs.IDs.GetAsInt64(i)
			score := rs.Scores[i]
			embedding, _ := rs.Fields.GetColumn(embeddingCol1).Get(i)

			log.Printf("ID: %d, score %f, embedding: %v\n", id, score, embedding)
		}
	}

	log.Println("start to execute hybrid search")

	result, err = c.HybridSearch(ctx, collectionName, nil, topK, []string{keyCol, embeddingCol1, embeddingCol2},
		client.NewRRFReranker(), []*client.ANNSearchRequest{
			client.NewANNSearchRequest(embeddingCol1, entity.L2, "", vec2search1, sp, topK),
			client.NewANNSearchRequest(embeddingCol2, entity.L2, "", vec2search2, sp, topK),
		})
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	log.Printf("hybrid search `%s` done, latency %v\n", collectionName, time.Since(begin))
	for _, rs := range result {
		for i := 0; i < rs.ResultCount; i++ {
			id, _ := rs.IDs.GetAsInt64(i)
			score := rs.Scores[i]
			embedding1, _ := rs.Fields.GetColumn(embeddingCol1).Get(i)
			embedding2, _ := rs.Fields.GetColumn(embeddingCol1).Get(i)
			log.Printf("ID: %d, score %f, embedding1: %v, embedding2: %v\n", id, score, embedding1, embedding2)
		}
	}

	c.DropCollection(ctx, collectionName)
}
