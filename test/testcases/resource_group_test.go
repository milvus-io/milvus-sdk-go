//go:build L3

package testcases

import (
	"context"
	"log"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

const configQnNodes = int32(4)
const newRgNode = int32(2)

func resetRgs(t *testing.T, ctx context.Context, mc *base.MilvusClient) {
	// release and drop all collections
	collections, _ := mc.ListCollections(ctx)
	for _, coll := range collections {
		mc.ReleaseCollection(ctx, coll.Name)
		err := mc.DropCollection(ctx, coll.Name)
		common.CheckErr(t, err, true)
	}

	// reset resource groups
	rgs, errList := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList, true)
	for _, rg := range rgs {
		if rg != common.DefaultRgName {
			// describe rg and get available node
			rgInfo, errDescribe := mc.DescribeResourceGroup(ctx, rg)
			common.CheckErr(t, errDescribe, true)

			// transfer available nodes into default rg
			if rgInfo.AvailableNodesNumber > 0 {
				errTransfer := mc.TransferNode(ctx, rg, common.DefaultRgName, rgInfo.AvailableNodesNumber)
				common.CheckErr(t, errTransfer, true)
			}

			// drop rg
			errDrop := mc.DropResourceGroup(ctx, rg)
			common.CheckErr(t, errDrop, true)
		}
	}

	rgs2, errList2 := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList2, true)
	require.Len(t, rgs2, 1)
}

// test rg default: list rg, create rg, describe rg, transfer node
func TestRgDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// describe default rg and check default rg info
	defaultRg, errDescribe := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDescribe, true)
	expDefaultRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes,
	}
	common.CheckResourceGroup(t, defaultRg, expDefaultRg)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// list rgs
	rgs, errList := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList, true)
	require.ElementsMatch(t, rgs, []string{common.DefaultRgName, rgName})

	// describe new rg and check new rg info
	myRg, errDescribe2 := mc.DescribeResourceGroup(ctx, rgName)
	common.CheckErr(t, errDescribe2, true)
	expRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             0,
		AvailableNodesNumber: 0,
	}
	common.CheckResourceGroup(t, myRg, expRg)

	// transfer node from default rg into new rg
	errTransfer := mc.TransferNode(ctx, common.DefaultRgName, rgName, newRgNode)
	common.CheckErr(t, errTransfer, true)

	// check rg info after transfer nodes
	myRg2, _ := mc.DescribeResourceGroup(ctx, rgName)
	transferRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             newRgNode,
		AvailableNodesNumber: newRgNode,
	}
	common.CheckResourceGroup(t, myRg2, transferRg)

	// check default rg info
	defaultRg2, _ := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	expDefaultRg2 := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes - newRgNode,
	}
	common.CheckResourceGroup(t, defaultRg2, expDefaultRg2)

	// try to drop default rg
	errDropDefault := mc.DropResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDropDefault, false, "delete default rg is not permitted")
}

// test create rg with invalid name
func TestCreateRgInvalidNames(t *testing.T) {
	t.Parallel()
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	type invalidNameStruct struct {
		name   string
		errMsg string
	}

	invalidNames := []invalidNameStruct{
		{name: "", errMsg: "resource group name couldn't be empty"},
		{name: "12-s", errMsg: "name must be an underscore or letter"},
		{name: "(mn)", errMsg: "name must be an underscore or letter"},
		{name: "中文", errMsg: "name must be an underscore or letter"},
		{name: "%$#", errMsg: "name must be an underscore or letter"},
		{name: common.GenLongString(common.MaxCollectionNameLen + 1), errMsg: "name must be less than 255 characters"},
	}
	// create rg with invalid name
	for _, invalidName := range invalidNames {
		errCreate := mc.CreateResourceGroup(ctx, invalidName.name)
		common.CheckErr(t, errCreate, false, invalidName.errMsg)
	}
}

// describe rg with not existed name
func TestDescribeRgNotExisted(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	_, errDescribe := mc.DescribeResourceGroup(ctx, common.GenRandomString(6))
	common.CheckErr(t, errDescribe, false, "resource group not found")
}

// drop rg with not existed name -> successfully
func TestDropRgNotExisted(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)

	errDrop := mc.DropResourceGroup(ctx, common.GenRandomString(6))
	common.CheckErr(t, errDrop, true)
}

// drop rg
func TestDropRgNonEmpty(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// transfer node
	errTransfer := mc.TransferNode(ctx, common.DefaultRgName, rgName, 1)
	common.CheckErr(t, errTransfer, true)

	// drop rg and rg available node is not 0
	errDrop := mc.DropResourceGroup(ctx, rgName)
	common.CheckErr(t, errDrop, false, "delete non-empty rg is not permitted")
}

// drop empty default rg
func TestDropEmptyRg(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// transfer node
	errTransfer := mc.TransferNode(ctx, common.DefaultRgName, rgName, configQnNodes)
	common.CheckErr(t, errTransfer, true)

	// describe default rg
	defaultRg, errDescribe := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDescribe, true)
	transferRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: 0,
	}
	common.CheckResourceGroup(t, defaultRg, transferRg)

	// drop empty default rg
	errDrop := mc.DropResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDrop, false, "delete default rg is not permitted")
}

// test list rgs
func TestListRgs(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create 10 new rgs
	rgNum := 10
	rgs := make([]string, 0, rgNum)
	for i := 1; i <= rgNum; i++ {
		rgName := common.GenRandomString(6)
		errCreate := mc.CreateResourceGroup(ctx, rgName)
		common.CheckErr(t, errCreate, true)
		rgs = append(rgs, rgName)
	}

	// list rgs
	listRgs, errList := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList, true)

	rgs = append(rgs, common.DefaultRgName)
	require.ElementsMatch(t, listRgs, rgs)
}

// test transfer node invalid number
func TestTransferInvalidNodes(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)
	type invalidNodesStruct struct {
		nodesNum int32
		errMsg   string
	}
	invalidNodes := []invalidNodesStruct{
		{nodesNum: 0, errMsg: "expected=NumNode > 0, actual=invalid NumNode"},
		{nodesNum: -1, errMsg: "expected=NumNode > 0, actual=invalid NumNode"},
		{nodesNum: 99, errMsg: "nodes not enough"},
	}
	// transfer node
	for _, invalidNode := range invalidNodes {
		errTransfer := mc.TransferNode(ctx, common.DefaultRgName, rgName, invalidNode.nodesNum)
		common.CheckErr(t, errTransfer, false, invalidNode.errMsg)
	}
}

// test transfer node rg not exist
func TestTransferRgNotExisted(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// source not exist
	errSource := mc.TransferNode(ctx, common.GenRandomString(6), common.DefaultRgName, newRgNode)
	common.CheckErr(t, errSource, false, "source resource group not found")

	// target not exist
	errTarget := mc.TransferNode(ctx, common.DefaultRgName, common.GenRandomString(6), newRgNode)
	common.CheckErr(t, errTarget, false, "target resource group not found")

	// transfer to self
	errSelf := mc.TransferNode(ctx, common.DefaultRgName, common.DefaultRgName, newRgNode)
	common.CheckErr(t, errSelf, true)

	defaultRg, _ := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	require.Equal(t, configQnNodes, defaultRg.AvailableNodesNumber)
}

// test transfer 2 replica2 from default to new rg
func TestTransferReplicas(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// transfer nodes into new rg
	errTransfer := mc.TransferNode(ctx, common.DefaultRgName, rgName, newRgNode)
	common.CheckErr(t, errTransfer, true)

	// load two replicas
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// load two replicas into default rg
	errLoad := mc.LoadCollection(ctx, collName, false, client.WithReplicaNumber(2), client.WithResourceGroups([]string{common.DefaultRgName}))
	common.CheckErr(t, errLoad, true)
	defaultRg, errDescribe := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDescribe, true)
	transferRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes - newRgNode,
		LoadedReplica:        map[string]int32{collName: 2},
	}
	common.CheckResourceGroup(t, defaultRg, transferRg)

	// transfer replica into new rg
	errReplica := mc.TransferReplica(ctx, common.DefaultRgName, rgName, collName, 2)
	common.CheckErr(t, errReplica, true)

	// check default rg
	defaultRg2, _ := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	transferRg2 := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes - newRgNode,
		IncomingNodeNum:      map[string]int32{collName: 2},
	}
	common.CheckResourceGroup(t, defaultRg2, transferRg2)

	// check new rg after transfer replica
	newRg, _ := mc.DescribeResourceGroup(ctx, rgName)
	expRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             newRgNode,
		AvailableNodesNumber: newRgNode,
		LoadedReplica:        map[string]int32{collName: 2},
		OutgoingNodeNum:      map[string]int32{collName: 2},
	}
	common.CheckResourceGroup(t, newRg, expRg)

	// drop new rg that loaded collection
	err := mc.DropResourceGroup(ctx, rgName)
	common.CheckErr(t, err, false, "some replicas still loaded in resource group")

	// search
	sp, err := entity.NewIndexHNSWSearchParam(74)
	searchRes, _ := mc.Search(
		ctx, collName,
		[]string{common.DefaultPartition},
		"",
		[]string{common.DefaultFloatFieldName},
		common.GenSearchVectors(common.DefaultNq, common.DefaultDim, entity.FieldTypeFloatVector),
		common.DefaultFloatVecFieldName,
		entity.L2,
		common.DefaultTopK,
		sp,
	)
	// check search result contains search vector, which from all partitions
	common.CheckErr(t, err, true)
	common.CheckSearchResult(t, searchRes, common.DefaultNq, common.DefaultTopK)

}

// test transfer replica of not existed collection
func TestTransferReplicaNotExistedCollection(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// transfer replica
	errTransfer := mc.TransferReplica(ctx, common.DefaultRgName, rgName, common.GenRandomString(3), 1)
	common.CheckErr(t, errTransfer, false, "can't find collection")
}

// test transfer replicas with invalid replica number
func TestTransferReplicaInvalidReplicaNumber(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)

	// create collection
	collName := createDefaultCollection(ctx, t, mc, false, 1)

	// invalid replicas
	type invalidReplicasStruct struct {
		replicaNumber int64
		errMsg        string
	}
	invalidReplicas := []invalidReplicasStruct{
		{replicaNumber: 0, errMsg: "expected=NumReplica > 0, actual=invalid NumReplica"},
		{replicaNumber: -1, errMsg: "expected=NumReplica > 0, actual=invalid NumReplica"},
		{replicaNumber: 1, errMsg: "only found [0] replicas in source resource group"},
	}

	for _, invalidReplica := range invalidReplicas {
		// transfer replica
		errTransfer := mc.TransferReplica(ctx, common.DefaultRgName, rgName, collName, invalidReplica.replicaNumber)
		common.CheckErr(t, errTransfer, false, invalidReplica.errMsg)
	}
}

// test transfer replicas rg not exist
func TestTransferReplicaRgNotExisted(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg and transfer nodes
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName)
	common.CheckErr(t, errCreate, true)
	mc.TransferNode(ctx, common.DefaultRgName, rgName, newRgNode)

	// init collection: create -> insert -> index -> load
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)
	mc.LoadCollection(ctx, collName, false, client.WithResourceGroups([]string{rgName}))

	// source not exist
	errSource := mc.TransferReplica(ctx, common.GenRandomString(6), common.DefaultRgName, collName, 1)
	common.CheckErr(t, errSource, false, "resource group not found")

	// target not exist
	errTarget := mc.TransferReplica(ctx, common.DefaultRgName, common.GenRandomString(6), collName, 1)
	common.CheckErr(t, errTarget, false, "resource group not found")

	// transfer to self -> error
	errSelf := mc.TransferReplica(ctx, rgName, rgName, collName, 1)
	common.CheckErr(t, errSelf, false, "replicas of same collection in target resource group")

	// transfer to default rg
	errTransfer := mc.TransferReplica(ctx, rgName, common.DefaultRgName, collName, 1)
	common.CheckErr(t, errTransfer, true)
	newRg, _ := mc.DescribeResourceGroup(ctx, rgName)
	log.Print(newRg)
	expRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             newRgNode,
		AvailableNodesNumber: newRgNode,
		IncomingNodeNum:      map[string]int32{collName: newRgNode},
	}
	common.CheckResourceGroup(t, newRg, expRg)
}
