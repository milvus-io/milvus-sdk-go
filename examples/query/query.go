package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr     = `localhost:19530`
	nEntities, dim = 3000, 8
	collectionName = "hello_milvus"

	msgFmt                         = "\n==== %s ====\n"
	idCol, randomCol, embeddingCol = "ID", "random", "embeddings"
	topK                           = 3
)

func main() {
	ctx := context.Background()

	fmt.Printf(msgFmt, "start connecting to Milvus")
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

	// define schema
	fmt.Printf(msgFmt, "create collection `hello_milvus")
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "hello_milvus is the simplest demo to introduce the APIs",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       idCol,
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       randomCol,
				DataType:   entity.FieldTypeDouble,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     embeddingCol,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: fmt.Sprintf("%d", dim),
				},
			},
		},
	}

	// create collection with consistency level, which serves as the default search/query consistency level
	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber, client.WithConsistencyLevel(entity.ClBounded)); err != nil {
		log.Fatalf("create collection failed, err: %v", err)
	}

	fmt.Printf(msgFmt, "start inserting random entities")
	idList, randomList := make([]int64, 0, nEntities), make([]float64, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	rand.Seed(time.Now().UnixNano())

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

	// build index
	fmt.Printf(msgFmt, "start creating index IVF_FLAT")
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		log.Fatalf("failed to create ivf flat index, err: %v", err)
	}
	if err := c.CreateIndex(ctx, collectionName, embeddingCol, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	// insert data
	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `hello_milvus, err: %v", err)
	}

	fmt.Printf(msgFmt, "start loading collection")
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	// search with default consistency level (bounded in this example)
	fmt.Printf(msgFmt, "start searcching based on vector similarity")
	vec2search := []entity.Vector{
		entity.FloatVector(embeddingList[len(embeddingList)-2]),
		entity.FloatVector(embeddingList[len(embeddingList)-1]),
	}
	begin := time.Now()
	sp, _ := entity.NewIndexFlatSearchParam()
	sRet, err := c.Search(ctx, collectionName, nil, "", []string{randomCol}, vec2search,
		embeddingCol, entity.L2, topK, sp)
	end := time.Now()
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	fmt.Println("results:")
	for _, res := range sRet {
		printResult(&res)
	}
	fmt.Printf("\tbounded consistency search latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// search with strong consistency level
	begin = time.Now()
	sp, _ = entity.NewIndexFlatSearchParam()
	sRet, err = c.Search(ctx, collectionName, nil, "", []string{randomCol}, vec2search,
		embeddingCol, entity.L2, topK, sp, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
	end = time.Now()
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	fmt.Println("results:")
	for _, res := range sRet {
		printResult(&res)
	}
	fmt.Printf("\tstrong consistency search latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// drop collection
	fmt.Printf(msgFmt, "drop collection `hello_milvus`")
	if err := c.DropCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to drop collection, err: %v", err)
	}
}

func printResult(sRet *client.SearchResult) {
	randoms := make([]float64, 0, sRet.ResultCount)
	scores := make([]float32, 0, sRet.ResultCount)

	var randCol *entity.ColumnDouble
	for _, field := range sRet.Fields {
		if field.Name() == randomCol {
			c, ok := field.(*entity.ColumnDouble)
			if ok {
				randCol = c
			}
		}
	}
	for i := 0; i < sRet.ResultCount; i++ {
		val, err := randCol.ValueByIdx(i)
		if err != nil {
			log.Fatal(err)
		}
		randoms = append(randoms, val)
		scores = append(scores, sRet.Scores[i])
	}
	fmt.Printf("\trandoms: %v, scores: %v\n", randoms, scores)
}
