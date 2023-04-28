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
	maxLength      = 10
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

	// create collection
	fmt.Printf(msgFmt, "create collection `hello_milvus")
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "hello_milvus is the simplest demo to introduce the APIs",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       idCol,
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{
					entity.TypeParamMaxLength: fmt.Sprintf("%d", maxLength),
				},
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

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil { // use server default shard number
		log.Fatalf("create collection failed, err: %v", err)
	}

	// insert data
	fmt.Printf(msgFmt, "start inserting random entities")
	idList := make([]string, 0, nEntities)
	randomList := make([]float64, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	rand.Seed(time.Now().UnixNano())

	// generate data
	for i := 0; i < nEntities; i++ {
		idList = append(idList, fmt.Sprintf("%d", i))
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

	idColData := entity.NewColumnVarChar(idCol, idList)
	randomColData := entity.NewColumnDouble(randomCol, randomList)
	embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)

	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `hello_milvus, err: %v", err)
	}

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}

	// build index
	fmt.Printf(msgFmt, "start creating index IVF_FLAT")
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		log.Fatalf("failed to create ivf flat index, err: %v", err)
	}
	if err := c.CreateIndex(ctx, collectionName, embeddingCol, idx, false); err != nil {
		log.Fatalf("failed to create index, err: %v", err)
	}

	fmt.Printf(msgFmt, "start loading collection")
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

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
	fmt.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// hybrid search
	fmt.Printf(msgFmt, "start hybrid searching with `random > 0.5`")
	begin = time.Now()
	sRet2, err := c.Search(ctx, collectionName, nil, "random > 0.5",
		[]string{randomCol}, vec2search, embeddingCol, entity.L2, topK, sp)
	end = time.Now()
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}
	fmt.Println("results:")
	for _, res := range sRet2 {
		printResult(&res)
	}
	fmt.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// delete data
	fmt.Printf(msgFmt, "start deleting with expr ``")
	pks := entity.NewColumnVarChar(idCol, []string{"0", "1"})
	sRet3, err := c.QueryByPks(ctx, collectionName, nil, pks, []string{randomCol})
	if err != nil {
		log.Fatalf("failed to query result, err: %v", err)
	}
	fmt.Println("results:")
	idlist := make([]string, 0)
	randList := make([]float64, 0)

	for _, col := range sRet3 {
		if col.Name() == idCol {
			idColumn := col.(*entity.ColumnVarChar)
			for i := 0; i < col.Len(); i++ {
				val, err := idColumn.ValueByIdx(i)
				if err != nil {
					log.Fatal(err)
				}
				idlist = append(idlist, val)
			}
		} else {
			randColumn := col.(*entity.ColumnDouble)
			for i := 0; i < col.Len(); i++ {
				val, err := randColumn.ValueByIdx(i)
				if err != nil {
					log.Fatal(err)
				}
				randList = append(randList, val)
			}
		}
	}
	fmt.Printf("\tids: %#v, randoms: %#v\n", idlist, randList)

	if err := c.DeleteByPks(ctx, collectionName, "", pks); err != nil {
		log.Fatalf("failed to delete by pks, err: %v", err)
	}
	_, err = c.QueryByPks(ctx, collectionName, nil, pks, []string{randomCol}, client.WithSearchQueryConsistencyLevel(entity.ClStrong))
	if err != nil {
		log.Printf("failed to query result, err: %v", err)
	}

	// drop collection
	fmt.Printf(msgFmt, "drop collection `hello_milvus`")
	if err := c.DropCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to drop collection, err: %v", err)
	}
}

func printResult(sRet *client.SearchResult) {
	pks := make([]string, 0, sRet.ResultCount)
	scores := make([]float32, 0, sRet.ResultCount)
	idCol, ok := sRet.IDs.(*entity.ColumnVarChar)
	if !ok {
		panic("not varchar pk")
	}
	for i := 0; i < sRet.ResultCount; i++ {
		val, err := idCol.ValueByIdx(i)
		if err != nil {
			log.Fatal(err)
		}
		pks = append(pks, val)
		scores = append(scores, sRet.Scores[i])
	}
	fmt.Printf("\tpks: %v, scores: %v\n", pks, scores)
}
