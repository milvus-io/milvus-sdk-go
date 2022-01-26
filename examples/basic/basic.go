package main

import (
	"context"
	"log"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// basic milvus operation example
func main() {
	// Milvus instance proxy address, may verify in your env/settings
	milvusAddr := `localhost:19530`

	// setup context for client creation, use 2 seconds here
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	c, err := client.NewGrpcClient(ctx, milvusAddr)
	if err != nil {
		// handling error and exit, to make example simple here
		log.Fatal("failed to connect to milvus:", err.Error())
	}

	collectionName := `gosdk_basic_collection`

	// first, lets check the collection exists
	collExists, err := c.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to check collection exists:", err.Error())
	}
	if collExists {
		// let's say the example collection is only for sampling the API
		// drop old one in case early crash or something
		_ = c.DropCollection(ctx, collectionName)
	}

	// define collection schema
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "this is the basic example collection",
		AutoID:         true,
		Fields: []*entity.Field{
			// currently primary key field is compulsory, and only int64 is allowd
			{
				Name:       "int64",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     true,
			},
			// also the vector field is needed
			{
				Name:     "vector",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{ // the vector dim may changed def method in release
					entity.TYPE_PARAM_DIM: "128",
				},
			},
		},
	}
	err = c.CreateCollection(ctx, schema, 2) // default shards num is 2
	if err != nil {
		log.Fatal("failed to create collection:", err.Error())
	}

	collections, err := c.ListCollections(ctx)
	if err != nil {
		log.Fatal("failed to list collections:", err.Error())
	}
	for _, collection := range collections {
		// print all the collections, id & name
		log.Printf("Collection id: %d, name: %s\n", collection.ID, collection.Name)
	}

	// show collection partitions
	partitions, err := c.ShowPartitions(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to show partitions:", err.Error())
	}
	for _, partition := range partitions {
		// print partition info, the shall be a default partition for out collection
		log.Printf("partition id: %d, name: %s\n", partition.ID, partition.Name)
	}

	partitionName := "new_partition"
	// now let's try to create a partition
	err = c.CreatePartition(ctx, collectionName, partitionName)
	if err != nil {
		log.Fatal("failed to create partition:", err.Error())
	}

	log.Println("After create partition")
	// show collection partitions, check creation
	partitions, err = c.ShowPartitions(ctx, collectionName)
	if err != nil {
		log.Fatal("failed to show partitions:", err.Error())
	}
	for _, partition := range partitions {
		log.Printf("partition id: %d, name: %s\n", partition.ID, partition.Name)
	}

	// clean up our mess
	_ = c.DropCollection(ctx, collectionName)
	c.Close()
}
