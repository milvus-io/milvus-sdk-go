package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc/credentials"
)

func main() {
	// Milvus instance proxy address, may verify in your env/settings
	milvusAddr := `localhost:19530`

	// setup context for client creation, use 2 seconds here
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	// setup tls
	cert, err := tls.LoadX509KeyPair("../cert/client.pem", "../cert/client.key") // 注意这里是 client 的证书
	if err != nil {
		log.Fatalf("tls.LoadX509KeyPair err: %v", err)
	}

	certPool := x509.NewCertPool()
	ca, err := os.ReadFile("../cert/ca.pem")
	if err != nil {
		log.Fatalf("ioutil.ReadFile err: %v", err)
	}

	if ok := certPool.AppendCertsFromPEM(ca); !ok {
		log.Fatalf("certPool.AppendCertsFromPEM err")
	}

	creds := credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{cert},
		ServerName:   "localhost",
		RootCAs:      certPool,
		MinVersion:   tls.VersionTLS13,
	})

	c, err := client.NewGrpcClient(ctx, milvusAddr, grpc.WithTransportCredentials(creds))
	if err != nil {
		// handling error and exit, to make example simple here
		log.Fatal("failed to connect to milvus:", err.Error())
	}
	// in a main func, remember to close the client
	defer c.Close()

	// here is the collection name we use in this example
	collectionName := `gosdk_insert_example`

	has, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to check whether collection exists:", err.Error())
	}
	if has {
		// collection with same name exist, clean up mess
		_ = c.DropCollection(ctx, collectionName)
	}

	// define collection schema, see film.csv
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "this is the example collection for inser and search",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "ID",
				DataType:   entity.FieldTypeInt64, // int64 only for now
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "Year",
				DataType:   entity.FieldTypeInt32,
				PrimaryKey: false,
				AutoID:     false,
			},
			{
				Name:     "Vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TypeParamDim: "8",
				},
			},
		},
	}

	err = c.CreateCollection(ctx, schema, 1) // only 1 shard
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
	vectors := make([][]float32, 0, len(films))
	// string field is not supported yet
	idTitle := make(map[int64]string)
	for idx, film := range films {
		ids = append(ids, film.ID)
		idTitle[film.ID] = film.Title
		years = append(years, film.Year)
		vectors = append(vectors, films[idx].Vector[:]) // prevent same vector
	}
	idColumn := entity.NewColumnInt64("ID", ids)
	yearColumn := entity.NewColumnInt32("Year", years)
	vectorColumn := entity.NewColumnFloatVector("Vector", 8, vectors)

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

	// load collection with async=false
	err = c.LoadCollection(ctx, collectionName, false)
	if err != nil {
		log.Fatal("failed to load collection:", err.Error())
	}
	log.Println("load collection completed")

	searchFilm := films[0] // use first fim to search
	vector := entity.FloatVector(searchFilm.Vector[:])
	// Use flat search param
	sp, _ := entity.NewIndexFlatSearchParam()
	sr, err := c.Search(ctx, collectionName, []string{}, "Year > 1990", []string{"ID"}, []entity.Vector{vector}, "Vector",
		entity.L2, 10, sp)
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
			log.Fatal("result field not math")
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
	Vector [8]float32 // fix length array
}

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
		if len(parts) != 8 { // dim must be 8
			continue
		}
		for idx, part := range parts {
			part = strings.TrimSpace(part)
			v, err := strconv.ParseFloat(part, 32)
			if err != nil {
				continue
			}
			fi.Vector[idx] = float32(v)
		}
		films = append(films, fi)
	}
	return films, nil
}
