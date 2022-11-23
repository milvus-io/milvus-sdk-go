package entity

import common "github.com/milvus-io/milvus-proto/go-api/commonpb"

// User is the model for RBAC user object.
type User struct {
	Name string
}

// Role is the model for RBAC role object.
type Role struct {
	Name string
}

// PriviledgeObjectType is an alias of common.ObjectType.
// used in RBAC related API.
type PriviledgeObjectType common.ObjectType

const (
	// PriviledegeObjectTypeCollection const value for collection.
	PriviledegeObjectTypeCollection PriviledgeObjectType = PriviledgeObjectType(common.ObjectType_Collection)
	// PriviledegeObjectTypeUser const value for user.
	PriviledegeObjectTypeUser PriviledgeObjectType = PriviledgeObjectType(common.ObjectType_User)
	// PriviledegeObjectTypeGlobal const value for Global.
	PriviledegeObjectTypeGlobal PriviledgeObjectType = PriviledgeObjectType(common.ObjectType_Global)
)
