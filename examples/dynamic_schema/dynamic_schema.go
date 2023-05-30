package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr     = `127.0.0.1:19530`
	nEntities, dim = 3000, 128
	collectionName = "dynamic_example"

	msgFmt                                             = "\n==== %s ====\n"
	idCol, typeCol, randomCol, sourceCol, embeddingCol = "ID", "type", "random", "source", "embeddings"
	topK                                               = 4
)

func main() {
	flag.Parse()
	ctx := context.Background()

	fmt.Printf(msgFmt, "start connecting to Milvus")
	c, err := client.NewGrpcClient(ctx, milvusAddr)
	if err != nil {
		log.Fatalf("failed to connect to milvus, err: %v", err)
	}
	defer c.Close()

	version, err := c.GetVersion(ctx)
	if err != nil {
		log.Fatal("failed to get version of Milvus server", err.Error())
	}
	fmt.Println("Milvus Version:", version)

	// delete collection if exists
	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("failed to check collection exists, err: %v", err)
	}
	if has {
		c.DropCollection(ctx, collectionName)
	}

	// create collection
	fmt.Printf(msgFmt, "create collection `dynamic_example")
	schema := entity.NewSchema().
		WithName(collectionName).
		WithDescription("dynamic schema example collection").
		WithAutoID(false).
		WithDynamicFieldEnabled(true).
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

	if err := c.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil { // use default shard number
		log.Fatalf("create collection failed, err: %v", err)
	}

	// describe collection
	fmt.Printf(msgFmt, "describe collection `dynamic_example`")
	coll, err := c.DescribeCollection(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to describe collection:", err.Error())
	}

	fmt.Printf("Collection %s\tDescription: %s\tDynamicEnabled: %t\n", coll.Schema.CollectionName, coll.Schema.CollectionName, coll.Schema.EnableDynamicField)
	for _, field := range coll.Schema.Fields {
		fmt.Printf("Field: %s\tDataType: %s\tIsDynamic: %t\n", field.Name, field.DataType.String(), field.IsDynamic)
	}

	// insert data
	fmt.Printf(msgFmt, "start inserting with extra columns")
	idList, randomList := make([]int64, 0, nEntities), make([]float64, 0, nEntities)
	typeList := make([]int32, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	rand.Seed(time.Now().UnixNano())
	// generate data
	for i := 0; i < nEntities; i++ {
		idList = append(idList, int64(i))
		typeList = append(typeList, int32(i%3))
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
	typeColData := entity.NewColumnInt32(typeCol, typeList)
	embeddingColData := entity.NewColumnFloatVector(embeddingCol, dim, embeddingList)
	if _, err := c.Insert(ctx, collectionName, "", idColData, randomColData, typeColData, embeddingColData); err != nil {
		log.Fatalf("failed to insert random data into `dynamic_example, err: %v", err)
	}

	if err := c.Flush(ctx, collectionName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}
	fmt.Printf(msgFmt, "start inserting with rows")

	// insert by struct
	type DynamicRow struct {
		ID     int64     `milvus:"name:ID;primary_key"`
		Vector []float32 `milvus:"name:embeddings;dim:128"`
		Source int32     `milvus:"name:source"`
		Value  float64   `milvus:"name:random"`
	}

	rows := make([]interface{}, 0, nEntities)
	for i := 0; i < nEntities; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}

		rows = append(rows, &DynamicRow{
			ID:     int64(nEntities + i),
			Vector: vec,
			Source: 1,
			Value:  rand.Float64(),
		})
	}

	_, err = c.InsertRows(ctx, collectionName, "", rows)
	if err != nil {
		log.Fatal("failed to insert by rows: ", err.Error())
	}

	// insert by map[string]interface{}
	fmt.Printf(msgFmt, "start to inserting by MapRow")
	m := make(map[string]interface{})
	m["ID"] = int64(nEntities)
	vec := make([]float32, 0, dim)
	for j := 0; j < dim; j++ {
		vec = append(vec, rand.Float32())
	}
	m["embeddings"] = vec
	m["source"] = int32(1)
	m["random"] = rand.Float64()

	_, err = c.InsertRows(ctx, collectionName, "", []interface{}{m})
	if err != nil {
		log.Fatal("failed to insert by rows: ", err.Error())
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
	start := time.Now()
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	fmt.Printf("load collection done, time elasped: %v\n", time.Since(start))
	fmt.Printf(msgFmt, "start searching based on vector similarity")

	vec2search := []entity.Vector{
		entity.FloatVector(embeddingList[len(embeddingList)-2]),
		entity.FloatVector(embeddingList[len(embeddingList)-1]),
	}
	begin := time.Now()
	sp, _ := entity.NewIndexIvfFlatSearchParam(16)
	sRet, err := c.Search(ctx, collectionName, nil, "", []string{randomCol, typeCol}, vec2search,
		embeddingCol, entity.L2, topK, sp)
	end := time.Now()
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}

	fmt.Println("results:")
	for _, res := range sRet {
		printResult(&res, map[string]string{randomCol: "double", typeCol: "int"})
	}
	fmt.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// hybrid search
	fmt.Printf(msgFmt, "start hybrid searching with `random > 0.9`")
	begin = time.Now()
	sRet2, err := c.Search(ctx, collectionName, nil, "random > 0.9",
		[]string{randomCol, typeCol}, vec2search, embeddingCol, entity.L2, topK, sp)
	end = time.Now()
	if err != nil {
		log.Fatalf("failed to search collection, err: %v", err)
	}
	fmt.Println("results:")
	for _, res := range sRet2 {
		printResult(&res, map[string]string{randomCol: "double", typeCol: "int"})
	}
	fmt.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)

	// query
	expr := "ID in [0, 1, 2]"
	fmt.Printf(msgFmt, fmt.Sprintf("query with expr `%s`", expr))
	sRet3, err := c.Query(ctx, collectionName, nil, expr, []string{randomCol, typeCol})
	if err != nil {
		log.Fatalf("failed to query result, err: %v", err)
	}
	printResultSet(sRet3, map[string]string{idCol: "int", randomCol: "double", typeCol: "int"})

	// $meta["source"]
	expr = "source in [1] and random > 0.1"
	fmt.Printf(msgFmt, fmt.Sprintf("query with expr `%s`", expr))
	sRet3, err = c.Query(ctx, collectionName, nil, expr, []string{randomCol, typeCol, sourceCol}, client.WithLimit(3))
	if err != nil {
		log.Fatalf("failed to query result, err: %v", err)
	}
	printResultSet(sRet3, map[string]string{idCol: "int", randomCol: "double", typeCol: "int", sourceCol: "int"})

	// with wildcard "*"
	fmt.Printf(msgFmt, fmt.Sprintf("query with expr `%s`, output = [`*`, `random`]", expr))
	sRet3, err = c.Query(ctx, collectionName, nil, expr, []string{"*", randomCol}, client.WithLimit(3))
	if err != nil {
		log.Printf("failed to query result with wildcast, err:%s\n", err.Error())
	}
	if sRet3 != nil {
		printResultSet(sRet3, map[string]string{randomCol: "double", "$meta": "json"})
	}

	// drop collection
	fmt.Printf(msgFmt, "drop collection `dynamic_example`")
	if err := c.DropCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to drop collection, err: %v", err)
	}
}

func printResultSet(sRets client.ResultSet, outputInfo map[string]string) {
	for name, typ := range outputInfo {
		column := sRets.GetColumn(name)
		if column == nil {
			fmt.Printf("column %s not found in result set\n", name)
			continue
		}

		fmt.Printf("Result Column %s, count: %d\n", name, column.Len())
		switch typ {
		case "int":
			var data []int64
			for i := 0; i < column.Len(); i++ {
				line, err := column.GetAsInt64(i)
				if err != nil {
					fmt.Printf("failed to get column %s at index %d, err: %s\n", name, i, err.Error())
				}
				data = append(data, line)
			}
			fmt.Println("Data:", data)
		case "string":
			var data []string
			for i := 0; i < column.Len(); i++ {
				line, err := column.GetAsString(i)
				if err != nil {
					fmt.Printf("failed to get column %s at index %d, err: %s\n", name, i, err.Error())
				}
				data = append(data, line)
			}
			fmt.Println("Data:", data)
		case "double":
			var data []float64
			for i := 0; i < column.Len(); i++ {
				line, err := column.GetAsDouble(i)
				if err != nil {
					fmt.Printf("failed to get column %s at index %d, err: %s\n", name, i, err.Error())
				}
				data = append(data, line)
			}
			fmt.Println("Data:", data)
		case "json":
			var data []string
			for i := 0; i < column.Len(); i++ {
				line, err := column.GetAsString(i)
				if err != nil {
					fmt.Printf("failed to get column %s at index %d, err: %s\n", name, i, err.Error())
				}
				data = append(data, line)
			}
			fmt.Println("Data:", data)
		}
	}
}

func printResult(sRet *client.SearchResult, outputInfo map[string]string) {
	printResultSet(sRet.Fields, outputInfo)
}
