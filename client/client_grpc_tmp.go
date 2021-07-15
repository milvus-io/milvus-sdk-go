package client

//// Search collection
//func (c *grpcClient) Search2(ctx context.Context, collName string) {
//	if c.service == nil {
//		return
//	}
//	req := &server.SearchRequest{
//		DbName:         "", // reserved
//		CollectionName: collName,
//		PartitionNames: []string{},
//		Dsl:            "", //TODO
//		DslType:        common.DslType_Dsl,
//		OutputFields:   []string{},
//	}
//	resp, err := c.service.Search(ctx, req)
//	if err != nil {
//		return
//	}
//	if err := handleRespStatus(resp.GetStatus()); err != nil {
//		return
//	}
//	//resp.Results.
//}
//
//func (c *grpcClient) SearchWithExpression() {
//
//}
//
//func (c *grpcClient) DslSearch(ctx context.Context, collName string, dsl string, outputFields []string, placeHolders entity.Column) {
//	if c.service == nil {
//		return
//	}
//	if placeHolders.Len() == 0 {
//		return
//	}
//	searchParams := entity.MapKvPairs(map[string]string{
//		"metric_type": "L2",
//		"nprobe":      "10",
//	})
//	req := &server.SearchRequest{
//		DbName:         "", // reserved
//		CollectionName: collName,
//		PartitionNames: []string{},
//		SearchParams:   searchParams,
//		Dsl:            dsl,
//		DslType:        common.DslType_Dsl,
//		OutputFields:   outputFields,
//	}
//	c.service.Search(ctx, req)
//}
