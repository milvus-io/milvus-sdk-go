package main

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	milvusAddr           = `localhost:19530`
	recycleResourceGroup = `__recycle_resource_group`
	defaultResourceGroup = `__default_resource_group`
	rg1                  = `rg1`
	rg2                  = `rg2`
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := client.NewClient(ctx, client.Config{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus, err: ", err.Error())
	}
	defer c.Close()

	ctx = context.Background()
	showAllResourceGroup(ctx, c)

	// query node count: 1
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 0 |
	// | rg1 | 0 | 0 | 0 |
	// | rg2 | 0 | 0 | 0 |
	if err := initializeCluster(ctx, c); err != nil {
		log.Fatal("failed to initialize cluster, err: ", err.Error())
	}

	showAllResourceGroup(ctx, c)

	// do some resource group managements.
	if err := resourceGroupManagement(ctx, c); err != nil {
		log.Fatal("failed to manage resource group, err: ", err.Error())
	}
}

// initializeCluster initializes the cluster with 4 resource groups.
func initializeCluster(ctx context.Context, c client.Client) error {
	// Use a huge resource group to hold the redundant query node.
	if err := c.CreateResourceGroup(ctx, recycleResourceGroup, client.WithCreateResourceGroupConfig(
		&entity.ResourceGroupConfig{
			Requests: &entity.ResourceGroupLimit{NodeNum: 0},
			Limits:   &entity.ResourceGroupLimit{NodeNum: 10000},
		},
	)); err != nil {
		return err
	}

	if err := c.UpdateResourceGroups(ctx, client.WithUpdateResourceGroupConfig(defaultResourceGroup, newResourceGroupCfg(1, 1))); err != nil {
		return err
	}

	if err := c.CreateResourceGroup(ctx, rg1); err != nil {
		return err
	}

	return c.CreateResourceGroup(ctx, rg2)
}

// resourceGroupManagement manages the resource groups.
func resourceGroupManagement(ctx context.Context, c client.Client) error {
	// Update resource group config.
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 0 |
	// | rg1 | 1 | 1 | 0 |
	// | rg2 | 2 | 2 | 0 |
	if err := c.UpdateResourceGroups(ctx,
		client.WithUpdateResourceGroupConfig(rg1, newResourceGroupCfg(1, 1)),
		client.WithUpdateResourceGroupConfig(rg2, newResourceGroupCfg(2, 2)),
	); err != nil {
		return err
	}
	showAllResourceGroup(ctx, c)

	// scale out cluster, new query node will be added to rg1 and rg2.
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 0 |
	// | rg1 | 1 | 1 | 1 |
	// | rg2 | 2 | 2 | 2 |
	scaleTo(ctx, 4)
	showAllResourceGroup(ctx, c)

	// scale out cluster, new query node will be added to __recycle__resource_group.
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 1 |
	// | rg1 | 1 | 1 | 1 |
	// | rg2 | 2 | 2 | 2 |
	scaleTo(ctx, 5)
	showAllResourceGroup(ctx, c)

	// Update resource group config, redundant query node will be transferred to recycle resource group.
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 2 |
	// | rg1 | 1 | 1 | 1 |
	// | rg2 | 1 | 1 | 1 |
	if err := c.UpdateResourceGroups(ctx,
		client.WithUpdateResourceGroupConfig(rg1, newResourceGroupCfg(1, 1)),
		client.WithUpdateResourceGroupConfig(rg2, newResourceGroupCfg(1, 1)),
	); err != nil {
		return err
	}
	showAllResourceGroup(ctx, c)

	// Update resource group config, rg1 and rg2 will transfer missing node from __recycle__resource_group.
	// | RG | Request | Limit | Nodes |
	// | -- | ------- | ----- | ----- |
	// | __default__resource_group | 1 | 1 | 1 |
	// | __recycle__resource_group | 0 | 10000 | 0 |
	// | rg1 | 2 | 2 | 2 |
	// | rg2 | 2 | 2 | 2 |
	if err := c.UpdateResourceGroups(ctx,
		client.WithUpdateResourceGroupConfig(rg1, newResourceGroupCfg(2, 2)),
		client.WithUpdateResourceGroupConfig(rg2, newResourceGroupCfg(2, 2)),
	); err != nil {
		return err
	}
	showAllResourceGroup(ctx, c)

	return nil
}

// scaleTo scales the cluster to the specified node number.
func scaleTo(_ context.Context, _ int) {
	// Cannot implement by milvus core and sdk,
	// Need to be implement by orchestration system.
}

func newResourceGroupCfg(request int32, limit int32) *entity.ResourceGroupConfig {
	return &entity.ResourceGroupConfig{
		Requests:     &entity.ResourceGroupLimit{NodeNum: request},
		Limits:       &entity.ResourceGroupLimit{NodeNum: limit},
		TransferFrom: []*entity.ResourceGroupTransfer{{ResourceGroup: recycleResourceGroup}},
		TransferTo:   []*entity.ResourceGroupTransfer{{ResourceGroup: recycleResourceGroup}},
	}
}

// showAllResourceGroup shows all resource groups.
func showAllResourceGroup(ctx context.Context, c client.Client) {
	rgs, err := c.ListResourceGroups(ctx)
	if err != nil {
		log.Fatal("failed to list resource groups, err: ", err.Error())
	}
	log.Println("resource groups:")
	for _, rg := range rgs {
		rg, err := c.DescribeResourceGroup(ctx, rg)
		if err != nil {
			log.Fatal("failed to describe resource group, err: ", err.Error())
		}
		results, err := json.Marshal(rg)
		if err != nil {
			log.Fatal("failed to marshal resource group, err: ", err.Error())
		}
		log.Printf("%s\n", results)
	}
}
