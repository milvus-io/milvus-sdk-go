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
	collectionName = "query_example"

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
	fmt.Printf(msgFmt, fmt.Sprintf("create collection `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription("query_example is a simple demo to demonstrate query API").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(randomCol).WithDataType(entity.FieldTypeDouble)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

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

	// insert data
	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `query_example`, err: %v", err)
	}

	// flush data
	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush collection `query_example`, err: %v", err)
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

	// search with strong consistency level
	// query
	expr := "ID in [0, 1, 2]"
	fmt.Printf(msgFmt, fmt.Sprintf("query with expr `%s`", expr))
	resultSet, err := c.Query(ctx, collectionName, nil, expr, []string{idCol, randomCol})
	if err != nil {
		log.Fatalf("failed to query result, err: %v", err)
	}
	printResultSet(resultSet, idCol, randomCol)

	// drop collection
	fmt.Printf(msgFmt, fmt.Sprintf("drop collection `%s`", collectionName))
	if err := c.DropCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to drop collection, err: %v", err)
	}
}

func printResultSet(rs client.ResultSet, outputFields ...string) {
	for _, fieldName := range outputFields {
		column := rs.GetColumn(fieldName)
		if column == nil {
			fmt.Printf("column %s not exists in result set\n", fieldName)
		}
		switch column.Type() {
		case entity.FieldTypeInt64:
			var result []int64
			for i := 0; i < column.Len(); i++ {
				v, err := column.GetAsInt64(i)
				if err != nil {
					fmt.Printf("column %s row %d cannot GetAsInt64, %s\n", fieldName, i, err.Error())
				}
				result = append(result, v)
			}
			fmt.Printf("Column %s: value: %v\n", fieldName, result)
		case entity.FieldTypeDouble:
			var result []float64
			for i := 0; i < column.Len(); i++ {
				v, err := column.GetAsDouble(i)
				if err != nil {
					fmt.Printf("column %s row %d cannot GetAsDouble, %s\n", fieldName, i, err.Error())
				}
				result = append(result, v)
			}
			fmt.Printf("Column %s: value: %v\n", fieldName, result)
		}
	}
}
