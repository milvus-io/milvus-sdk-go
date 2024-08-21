package main

import (
	"context"
	"log"

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

	// create user
	err = c.CreateCredential(ctx, "user123", "passwd1")
	if err != nil {
		log.Fatalf("failed to create user, err: %v", err)
	}

	// create role
	c.CreateRole(ctx, "role123")
	if err != nil {
		log.Fatalf("failed to create role, err: %v", err)
	}
	c.Grant(ctx, "role123", entity.PriviledegeObjectTypeGlobal, "*", "read")

	// grant role to user
	c.AddUserRole(ctx, "user123", "role123")

	// backup rbac
	meta, err := c.BackupRBAC(ctx)
	if err != nil {
		log.Fatalf("failed to backup rbac, err: %v", err)
	}

	// clean rbac to make restore works
	c.DropRole(ctx, "role123")
	c.DeleteCredential(ctx, "user123")
	c.Revoke(ctx, "role123", entity.PriviledegeObjectTypeGlobal, "*", "read")

	// restore rbac
	err = c.RestoreRBAC(ctx, meta)
	if err != nil {
		log.Fatalf("failed to restore rbac, err: %v", err)
	}

	// clean rbac
	c.DropRole(ctx, "role123")
	c.DeleteCredential(ctx, "user123")
	c.Revoke(ctx, "role123", entity.PriviledegeObjectTypeGlobal, "*", "read")
}
