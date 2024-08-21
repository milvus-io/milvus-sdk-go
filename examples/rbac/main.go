package main

import (
	"context"
	"log"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/utils/crypto"
	"google.golang.org/grpc/metadata"
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
		Address:  milvusAddr,
		Username: "root",
		Password: "Milvus",
	})
	if err != nil {
		log.Fatalf("failed to connect to milvus, err: %v", err)
	}
	defer c.Close()

	// clean rbac
	c.Revoke(ctx, "role123", entity.PriviledegeObjectTypeCollection, "*", "*")
	c.DeleteCredential(ctx, "user123")
	c.DropRole(ctx, "role123")

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

	grants, _ := c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))

	err = c.Grant(ctx, "role123", entity.PriviledegeObjectTypeCollection, "*", "Search")
	if err != nil {
		log.Fatalf("failed to grant role, err: %v", err)
	}
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))

	// grant role to user
	c.AddUserRole(ctx, "user123", "role123")
	c.AddUserRole(ctx, "user123", "public")
	c.AddUserRole(ctx, "user123", "admin")

	// backup rbac
	meta, err := c.BackupRBAC(ctx)
	if err != nil {
		log.Fatalf("failed to backup rbac, err: %v", err)
	}
	log.Println("user num: ", len(meta.Users))
	for _, user := range meta.Users {
		log.Println("user's role", user.Roles)
	}
	log.Println("role num: ", len(meta.Roles))
	log.Println("grants num: ", len(meta.RoleGrants))

	// clean rbac to make restore works
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))
	err = c.Revoke(ctx, "role123", entity.PriviledegeObjectTypeCollection, "*", "Search")
	if err != nil {
		log.Fatalf("failed to revoke, err: %v", err)
	}
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))
	c.DeleteCredential(ctx, "user123")
	err = c.DropRole(ctx, "role123")
	if err != nil {
		log.Fatalf("failed to drop role, err: %v", err)
	}

	log.Println("-----start to restore rbac-----")

	// restore rbac
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))

	err = c.RestoreRBAC(ctx, meta)
	if err != nil {
		log.Fatalf("failed to restore rbac, err: %v", err)
	}

	// backup rbac to check
	log.Println("-----verify restore result-----")
	meta, err = c.BackupRBAC(ctx)
	if err != nil {
		log.Fatalf("failed to backup rbac, err: %v", err)
	}
	log.Println("user num: ", len(meta.Users))
	for _, user := range meta.Users {
		log.Println("user's role", user.Roles)
	}
	log.Println("role num: ", len(meta.Roles))
	log.Println("grants num: ", len(meta.RoleGrants))

	// clean rbac
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))
	err = c.Revoke(ctx, "role123", entity.PriviledegeObjectTypeCollection, "*", "Search")
	if err != nil {
		log.Fatalf("failed to revoke, err: %v", err)
	}
	grants, _ = c.ListGrants(ctx, "role123", "default")
	log.Println("grants: ", len(grants))
	c.DeleteCredential(ctx, "user123")
	err = c.DropRole(ctx, "role123")
	if err != nil {
		log.Fatalf("failed to drop role, err: %v", err)
	}
}

func GetContext(ctx context.Context, originValue string) context.Context {
	authKey := strings.ToLower("authorization")
	authValue := crypto.Base64Encode(originValue)
	contextMap := map[string]string{
		authKey: authValue,
	}
	md := metadata.New(contextMap)
	return metadata.NewIncomingContext(ctx, md)
}
