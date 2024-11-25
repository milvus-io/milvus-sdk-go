//go:build L3

package testcases

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

const (
	configQnNodes = int32(4)
	newRgNode     = int32(2)
)

func resetRgs(t *testing.T, ctx context.Context, mc *base.MilvusClient) {
	t.Helper()
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
		//if rg != common.DefaultRgName {
		// describe rg and get available node
		err := mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rg, &entity.ResourceGroupConfig{
			Requests:     &entity.ResourceGroupLimit{NodeNum: 0},
			Limits:       &entity.ResourceGroupLimit{NodeNum: 0},
			TransferFrom: []*entity.ResourceGroupTransfer{},
			TransferTo:   []*entity.ResourceGroupTransfer{},
		}))
		common.CheckErr(t, err, true)
		//}
	}
	for _, rg := range rgs {
		if rg != common.DefaultRgName {
			errDrop := mc.DropResourceGroup(ctx, rg)
			common.CheckErr(t, errDrop, true)
		}
	}

	rgs2, errList2 := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList2, true)
	require.Len(t, rgs2, 1)
}

// No need for now
func onCheckRgAvailable(t *testing.T, ctx context.Context, mc *base.MilvusClient, rgName string, expAvailable int32) {
	for {
		select {
		case <-ctx.Done():
			require.FailNowf(t, "Available nodes cannot reach within timeout", "expAvailableNodeNum: %d", expAvailable)
		default:
			rg, _ := mc.DescribeResourceGroup(ctx, rgName)
			if int32(len(rg.Nodes)) == expAvailable {
				return
			}
			time.Sleep(time.Second * 2)
		}
	}
}

func checkResourceGroup(t *testing.T, ctx context.Context, mc *base.MilvusClient, expRg *entity.ResourceGroup) {
	actualRg, err := mc.DescribeResourceGroup(ctx, expRg.Name)
	common.CheckErr(t, err, true)
	common.CheckResourceGroup(t, actualRg, expRg)
}

// test rg default: list rg, create rg, describe rg, transfer node
func TestRgDefault(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// describe default rg and check default rg info: Limits: 1000000, Nodes: all
	expDefaultRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes,
		Config: &entity.ResourceGroupConfig{
			Limits: &entity.ResourceGroupLimit{NodeNum: 0},
		},
	}
	checkResourceGroup(t, ctx, mc, expDefaultRg)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	common.CheckErr(t, errCreate, true)

	// list rgs
	rgs, errList := mc.ListResourceGroups(ctx)
	common.CheckErr(t, errList, true)
	require.ElementsMatch(t, rgs, []string{common.DefaultRgName, rgName})

	// describe new rg and check new rg info
	expRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             newRgNode,
		AvailableNodesNumber: newRgNode,
		Config: &entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
			Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
		},
	}
	checkResourceGroup(t, ctx, mc, expRg)

	// update resource group
	errUpdate := mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rgName, &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: configQnNodes},
		Limits:   &entity.ResourceGroupLimit{NodeNum: configQnNodes},
	}))
	common.CheckErr(t, errUpdate, true)

	// check rg info after transfer nodes
	transferRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             configQnNodes,
		AvailableNodesNumber: configQnNodes,
		Config: &entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: configQnNodes},
		},
	}
	checkResourceGroup(t, ctx, mc, transferRg)

	// check default rg info: 0 Nodes
	expDefaultRg2 := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: 0,
		Config: &entity.ResourceGroupConfig{
			Limits: &entity.ResourceGroupLimit{NodeNum: 0},
		},
	}
	checkResourceGroup(t, ctx, mc, expDefaultRg2)

	// try to drop default rg
	errDropDefault := mc.DropResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDropDefault, false, "default resource group is not deletable")
}

func TestCreateRgWithTransfer(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create rg0 with requests=2, limits=3, total 4 nodes
	rg0 := common.GenRandomString(6)
	rg0Limits := newRgNode + 1
	errCreate := mc.CreateResourceGroup(ctx, rg0, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: rg0Limits},
	}))
	common.CheckErr(t, errCreate, true)

	// check rg0 available node: 3, default available node: 1
	actualRg0, _ := mc.DescribeResourceGroup(ctx, rg0)
	require.Lenf(t, actualRg0.Nodes, int(rg0Limits), fmt.Sprintf("expected %s has %d available nodes", rg0, rg0Limits))
	actualRgDef, _ := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	require.Lenf(t, actualRgDef.Nodes, int(configQnNodes-rg0Limits), fmt.Sprintf("expected %s has %d available nodes", common.DefaultRgName, int(configQnNodes-rg0Limits)))

	// create rg1 with TransferFrom & TransferTo & requests=3, limits=4
	rg1 := common.GenRandomString(6)
	rg1Requests := newRgNode + 1
	errCreate = mc.CreateResourceGroup(ctx, rg1, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: rg1Requests},
		Limits:   &entity.ResourceGroupLimit{NodeNum: configQnNodes},
		TransferFrom: []*entity.ResourceGroupTransfer{
			{ResourceGroup: rg0},
		},
		TransferTo: []*entity.ResourceGroupTransfer{
			{ResourceGroup: common.DefaultRgName},
		},
	}))

	// verify available nodes: rg0 + rg1 = configQnNodes = 4
	time.Sleep(time.Duration(rand.Intn(6)) * time.Second)
	actualRg0, _ = mc.DescribeResourceGroup(ctx, rg0)
	actualRg1, _ := mc.DescribeResourceGroup(ctx, rg1)
	require.EqualValuesf(t, configQnNodes, len(actualRg0.Nodes)+len(actualRg1.Nodes), fmt.Sprintf("Expected the total available nodes of %s and %s is %d ", rg0, rg1, configQnNodes))
	expDefaultRg1 := &entity.ResourceGroup{
		Name:                 rg1,
		Capacity:             rg1Requests,
		AvailableNodesNumber: -1, // not check
		Config: &entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: rg1Requests},
			Limits:   &entity.ResourceGroupLimit{NodeNum: configQnNodes},
			TransferFrom: []*entity.ResourceGroupTransfer{
				{ResourceGroup: rg0},
			},
			TransferTo: []*entity.ResourceGroupTransfer{
				{ResourceGroup: common.DefaultRgName},
			},
		},
	}
	checkResourceGroup(t, ctx, mc, expDefaultRg1)
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

func TestCreateUpdateRgWithNotExistTransfer(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create/update rg with not existed TransferFrom rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
		TransferFrom: []*entity.ResourceGroupTransfer{
			{ResourceGroup: "aaa"},
		},
	}))
	common.CheckErr(t, errCreate, false, "resource group in `TransferFrom` aaa not exist")
	errUpdate := mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rgName, &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
		TransferFrom: []*entity.ResourceGroupTransfer{
			{ResourceGroup: "aaa"},
		},
	}))
	common.CheckErr(t, errUpdate, false, "resource group not found")

	// create/update rg with not existed TransferTo rg
	errCreate = mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
		TransferTo: []*entity.ResourceGroupTransfer{
			{ResourceGroup: "aaa"},
		},
	}))
	common.CheckErr(t, errCreate, false, "resource group in `TransferTo` aaa not exist")
	errUpdate = mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rgName, &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
		TransferTo: []*entity.ResourceGroupTransfer{
			{ResourceGroup: "aaa"},
		},
	}))
	common.CheckErr(t, errUpdate, false, "resource group not found")
}

func TestCreateRgWithRequestsLimits(t *testing.T) {
	type requestsLimits struct {
		requests  int32
		limits    int32
		available int32
		errMsg    string
	}
	reqAndLimits := []requestsLimits{
		{requests: 0, limits: 0, available: 0},
		{requests: -1, limits: 0, errMsg: "node num in `requests` or `limits` should not less than 0"},
		{requests: 0, limits: -2, errMsg: "node num in `requests` or `limits` should not less than 0"},
		{requests: 10, limits: 1, errMsg: "limits node num should not less than requests node num"},
		{requests: 2, limits: 3, available: 3},
		{requests: configQnNodes * 2, limits: configQnNodes * 3, available: configQnNodes},
		{requests: configQnNodes, limits: configQnNodes, available: configQnNodes},
	}
	// connect
	ctx := createContext(t, time.Second*20)
	mc := createMilvusClient(ctx, t)

	rgName := common.GenRandomString(6)
	err := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Limits: &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	common.CheckErr(t, err, false, "requests or limits is required")

	for _, rl := range reqAndLimits {
		log.Println(rl)
		rgName := common.GenRandomString(6)
		resetRgs(t, ctx, mc)
		errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: rl.requests},
			Limits:   &entity.ResourceGroupLimit{NodeNum: rl.limits},
		}))
		if rl.errMsg != "" {
			common.CheckErr(t, errCreate, false, rl.errMsg)
		} else {
			expDefaultRg := &entity.ResourceGroup{
				Name:                 rgName,
				Capacity:             rl.requests,
				AvailableNodesNumber: rl.available,
				Config: &entity.ResourceGroupConfig{
					Requests: &entity.ResourceGroupLimit{NodeNum: rl.requests},
					Limits:   &entity.ResourceGroupLimit{NodeNum: rl.limits},
				},
			}
			checkResourceGroup(t, ctx, mc, expDefaultRg)
			// check available node
			//onDescribeRg(t, ctx, mc, rgName, rl.available)
		}
	}
}

func TestUpdateRgWithRequestsLimits(t *testing.T) {
	type requestsLimits struct {
		requests  int32
		limits    int32
		available int32
		errMsg    string
	}
	reqAndLimits := []requestsLimits{
		{requests: 0, limits: 0, available: 0},
		{requests: -1, limits: 0, errMsg: "node num in `requests` or `limits` should not less than 0"},
		{requests: 0, limits: -2, errMsg: "node num in `requests` or `limits` should not less than 0"},
		{requests: 10, limits: 1, errMsg: "limits node num should not less than requests node num"},
		{requests: 2, limits: 3, available: 3},
		{requests: configQnNodes * 2, limits: configQnNodes * 3, available: configQnNodes},
		{requests: configQnNodes, limits: configQnNodes, available: configQnNodes},
	}
	// connect
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	rgName := common.GenRandomString(6)
	err := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	err = mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rgName, &entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	common.CheckErr(t, err, false, "requests or limits is required")

	for _, rl := range reqAndLimits {
		log.Println(rl)
		errCreate := mc.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(rgName, &entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: rl.requests},
			Limits:   &entity.ResourceGroupLimit{NodeNum: rl.limits},
		}))
		if rl.errMsg != "" {
			common.CheckErr(t, errCreate, false, rl.errMsg)
		} else {
			expDefaultRg := &entity.ResourceGroup{
				Name:                 rgName,
				Capacity:             rl.requests,
				AvailableNodesNumber: rl.available,
				Config: &entity.ResourceGroupConfig{
					Requests: &entity.ResourceGroupLimit{NodeNum: rl.requests},
					Limits:   &entity.ResourceGroupLimit{NodeNum: rl.limits},
				},
			}
			checkResourceGroup(t, ctx, mc, expDefaultRg)
			// check available node
			//onDescribeRg(t, ctx, mc, rgName, rl.available)
		}
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
	errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	common.CheckErr(t, errCreate, true)

	// drop rg and rg available node is not 0
	errDrop := mc.DropResourceGroup(ctx, rgName)
	common.CheckErr(t, errDrop, false, "resource group's limits node num is not 0")
}

// drop empty default rg
func TestDropEmptyRg(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: configQnNodes},
		Limits:   &entity.ResourceGroupLimit{NodeNum: configQnNodes},
	}))
	common.CheckErr(t, errCreate, true)

	// describe default rg
	transferRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: 0,
	}
	checkResourceGroup(t, ctx, mc, transferRg)

	// drop empty default rg
	errDrop := mc.DropResourceGroup(ctx, common.DefaultRgName)
	common.CheckErr(t, errDrop, false, "default resource group is not deletable")
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
		{nodesNum: 0, errMsg: "invalid parameter[expected=NumNode > 0][actual=invalid NumNode 0]"},
		{nodesNum: -1, errMsg: "invalid parameter[expected=NumNode > 0][actual=invalid NumNode -1]"},
		{nodesNum: 99, errMsg: "resource group node not enough"},
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
	common.CheckErr(t, errSource, false, "resource group not found")

	// target not exist
	errTarget := mc.TransferNode(ctx, common.DefaultRgName, common.GenRandomString(6), newRgNode)
	common.CheckErr(t, errTarget, false, "resource group not found")

	// transfer to self
	errSelf := mc.TransferNode(ctx, common.DefaultRgName, common.DefaultRgName, newRgNode)
	common.CheckErr(t, errSelf, false, "source resource group and target resource group should not be the same")

	defaultRg, _ := mc.DescribeResourceGroup(ctx, common.DefaultRgName)
	require.Equal(t, configQnNodes, defaultRg.AvailableNodesNumber)
}

// test transfer 2 replica2 from default to new rg
func TestTransferReplicas(t *testing.T) {
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	// connect
	mc := createMilvusClient(ctx, t)
	resetRgs(t, ctx, mc)

	// create new rg with requests 2
	rgName := common.GenRandomString(6)
	errCreate := mc.CreateResourceGroup(ctx, rgName, client.WithCreateResourceGroupConfig(&entity.ResourceGroupConfig{
		Requests: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		Limits:   &entity.ResourceGroupLimit{NodeNum: newRgNode},
	}))
	common.CheckErr(t, errCreate, true)

	// load two replicas
	collName, _ := createCollectionWithDataIndex(ctx, t, mc, true, true)

	// load two replicas into default rg
	errLoad := mc.LoadCollection(ctx, collName, false, client.WithReplicaNumber(2), client.WithResourceGroups([]string{common.DefaultRgName}))
	common.CheckErr(t, errLoad, true)
	transferRg := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes - newRgNode,
		LoadedReplica:        map[string]int32{collName: 2},
		Config: &entity.ResourceGroupConfig{
			Limits: &entity.ResourceGroupLimit{NodeNum: 0},
		},
	}
	checkResourceGroup(t, ctx, mc, transferRg)

	// transfer replica into new rg
	errReplica := mc.TransferReplica(ctx, common.DefaultRgName, rgName, collName, 2)
	common.CheckErr(t, errReplica, true)

	// check default rg
	transferRg2 := &entity.ResourceGroup{
		Name:                 common.DefaultRgName,
		Capacity:             common.DefaultRgCapacity,
		AvailableNodesNumber: configQnNodes - newRgNode,
		IncomingNodeNum:      map[string]int32{collName: 2},
		Config: &entity.ResourceGroupConfig{
			Limits: &entity.ResourceGroupLimit{NodeNum: 0},
		},
	}
	checkResourceGroup(t, ctx, mc, transferRg2)

	// check new rg after transfer replica
	expRg := &entity.ResourceGroup{
		Name:                 rgName,
		Capacity:             newRgNode,
		AvailableNodesNumber: newRgNode,
		LoadedReplica:        map[string]int32{collName: 2},
		OutgoingNodeNum:      map[string]int32{collName: 2},
		Config: &entity.ResourceGroupConfig{
			Limits: &entity.ResourceGroupLimit{NodeNum: newRgNode},
		},
	}
	checkResourceGroup(t, ctx, mc, expRg)

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
	common.CheckErr(t, errTransfer, false, "collection not found")
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
	collName := createDefaultCollection(ctx, t, mc, false, common.DefaultShards)

	// invalid replicas
	type invalidReplicasStruct struct {
		replicaNumber int64
		errMsg        string
	}
	invalidReplicas := []invalidReplicasStruct{
		{replicaNumber: 0, errMsg: "invalid parameter[expected=NumReplica > 0][actual=invalid NumReplica 0]"},
		{replicaNumber: -1, errMsg: "invalid parameter[expected=NumReplica > 0][actual=invalid NumReplica -1]"},
		{replicaNumber: 1, errMsg: "Collection not loaded"},
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
	common.CheckErr(t, errSelf, false, "source resource group and target resource group should not be the same")

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
	checkResourceGroup(t, ctx, mc, expRg)
}
