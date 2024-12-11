package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func main() {
	// Milvus instance proxy address, may verify in your env/settings
	milvusAddr := `localhost:19530`

	// setup context for client creation, use 2 seconds here
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	c, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		// handling error and exit, to make example simple here
		log.Fatal("failed to connect to milvus:", err.Error())
	}
	// in a main func, remember to close the client
	defer c.Close()

	// here is the collection name we use in this example
	collectionName := `gosdk_sparse_example`

	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to check whether collection exists:", err.Error())
	}
	if has {
		// collection with same name exist, clean up mess
		_ = c.DropCollection(ctx, collectionName)
	}

	// define collection schema, see film.csv
	schema := entity.NewSchema().WithName(collectionName).WithDescription("this is the example collection for sparse vector").
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
		WithField(entity.NewField().WithName("Year").WithDataType(entity.FieldTypeInt32)).
		WithField(entity.NewField().WithName("Vector").WithDataType(entity.FieldTypeSparseVector))

	err = c.CreateCollection(ctx, schema, entity.DefaultShardNumber) // only 1 shard
	if err != nil {
		log.Fatal("failed to create collection:", err.Error())
	}

	films, err := loadFilmCSV()
	if err != nil {
		log.Fatal("failed to load film data csv:", err.Error())
	}

	// row-base covert to column-base
	ids := make([]int64, 0, len(films))
	years := make([]int32, 0, len(films))
	vectors := make([]entity.SparseEmbedding, 0, len(films))
	// string field is not supported yet
	idTitle := make(map[int64]string)
	for idx, film := range films {
		ids = append(ids, film.ID)
		idTitle[film.ID] = film.Title
		years = append(years, film.Year)
		vectors = append(vectors, films[idx].Vector)
	}
	idColumn := entity.NewColumnInt64("ID", ids)
	yearColumn := entity.NewColumnInt32("Year", years)
	vectorColumn := entity.NewColumnSparseVectors("Vector", vectors)

	// insert into default partition
	_, err = c.Insert(ctx, collectionName, "", idColumn, yearColumn, vectorColumn)
	if err != nil {
		log.Fatal("failed to insert film data:", err.Error())
	}
	log.Println("insert completed")
	ctx, cancel = context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	err = c.Flush(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to flush collection:", err.Error())
	}
	log.Println("flush completed")

	// Now add index
	idx, err := entity.NewIndexSparseInverted(entity.IP, 0.3)
	if err != nil {
		log.Fatal("fail to create sparse inverted index:", err.Error())
	}
	err = c.CreateIndex(ctx, collectionName, "Vector", idx, false)
	if err != nil {
		log.Fatal("fail to create index:", err.Error())
	}

	// load collection with async=false
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to load collection:", err.Error())
	}
	log.Println("load collection completed")

	searchFilm := films[0].Vector // use first fim to search
	// Use flat search param
	sp, _ := entity.NewIndexSparseInvertedSearchParam(0)
	sr, err := c.Search(ctx, collectionName, []string{}, "Year > 1990", []string{"ID"}, []entity.Vector{searchFilm}, "Vector",
		entity.IP, 10, sp)
	if err != nil {
		log.Fatal("fail to search collection:", err.Error())
	}
	for _, result := range sr {
		var idColumn *entity.ColumnInt64
		for _, field := range result.Fields {
			if field.Name() == "ID" {
				c, ok := field.(*entity.ColumnInt64)
				if ok {
					idColumn = c
				}
			}
		}
		if idColumn == nil {
			log.Fatal("result field not match")
		}
		for i := 0; i < result.ResultCount; i++ {
			id, err := idColumn.ValueByIdx(i)
			if err != nil {
				log.Fatal(err.Error())
			}
			title := idTitle[id]
			fmt.Printf("file id: %d title: %s scores: %f\n", id, title, result.Scores[i])
		}
	}

	// clean up
	_ = c.DropCollection(ctx, collectionName)
}

type film struct {
	ID     int64
	Title  string
	Year   int32
	Vector entity.SparseEmbedding
}

// unlike other examples, this converts the vector field into a random sparse vector
func loadFilmCSV() ([]film, error) {
	f, err := os.Open("../films.csv") // assume you are in examples/insert folder, if not, please change the path
	if err != nil {
		return []film{}, err
	}
	r := csv.NewReader(f)
	raw, err := r.ReadAll()
	if err != nil {
		return []film{}, err
	}
	films := make([]film, 0, len(raw))
	for _, line := range raw {
		if len(line) < 4 { // insuffcient column
			continue
		}
		fi := film{}
		// ID
		v, err := strconv.ParseInt(line[0], 10, 64)
		if err != nil {
			continue
		}
		fi.ID = v
		// Title
		fi.Title = line[1]
		// Year
		v, err = strconv.ParseInt(line[2], 10, 64)
		if err != nil {
			continue
		}
		fi.Year = int32(v)
		// Vector
		vectorStr := strings.ReplaceAll(line[3], "[", "")
		vectorStr = strings.ReplaceAll(vectorStr, "]", "")
		parts := strings.Split(vectorStr, ",")

		sparsePositions := make([]uint32, 0, len(parts))
		// randomly pick some uint32 as dimension
		uniquePositions := make(map[uint32]struct{})
		for len(uniquePositions) < len(parts) {
			uniquePositions[uint32(rand.Intn(1000))] = struct{}{}
		}
		for k := range uniquePositions {
			sparsePositions = append(sparsePositions, k)
		}

		sparseValues := make([]float32, 0, len(parts))
		for _, part := range parts {
			part = strings.TrimSpace(part)
			v, err := strconv.ParseFloat(part, 32)
			if err != nil {
				continue
			}
			sparseValues = append(sparseValues, float32(v))
		}
		fi.Vector, err = entity.NewSliceSparseEmbedding(sparsePositions, sparseValues)
		if err != nil {
			return nil, err
		}
		films = append(films, fi)
	}
	return films, nil
}
