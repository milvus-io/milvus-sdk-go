package entity

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/rgpb"
)

type (
	ResourceGroupConfig   = rgpb.ResourceGroupConfig
	ResourceGroupLimit    = rgpb.ResourceGroupLimit
	ResourceGroupTransfer = rgpb.ResourceGroupTransfer
	NodeInfo              = commonpb.NodeInfo
)

// ResourceGroup information model struct.
type ResourceGroup struct {
	Name                 string
	Capacity             int32
	AvailableNodesNumber int32
	LoadedReplica        map[string]int32
	OutgoingNodeNum      map[string]int32
	IncomingNodeNum      map[string]int32
	Config               *ResourceGroupConfig
	Nodes                []*NodeInfo
}
