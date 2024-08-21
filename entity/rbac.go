package entity

import (
	common "github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
)

// User is the model for RBAC user object.
type User struct {
	Name string
}

// UserDescription is the model for RBAC user description object.
type UserDescription struct {
	Name  string
	Roles []string
}

// RoleGrants is the model for RBAC role description object.
type RoleGrants struct {
	Object        string
	ObjectName    string
	RoleName      string
	GrantorName   string
	PrivilegeName string
	DbName        string
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

type OperatePrivilegeOpt struct {
	Base     *common.MsgBase
	Database string
}

type OperatePrivilegeOption func(o *OperatePrivilegeOpt)

func WithOperatePrivilegeBase(base *common.MsgBase) OperatePrivilegeOption {
	return func(o *OperatePrivilegeOpt) {
		o.Base = base
	}
}

func WithOperatePrivilegeDatabase(database string) OperatePrivilegeOption {
	return func(o *OperatePrivilegeOpt) {
		o.Database = database
	}
}

type UserInfo struct {
	UserDescription
	Password string
}

type RBACMeta struct {
	Users      []*UserInfo
	Roles      []*Role
	RoleGrants []*RoleGrants
}
