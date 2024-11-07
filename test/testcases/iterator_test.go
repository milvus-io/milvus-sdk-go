//go:build L0

package testcases

// func TestSearchIteratorDefault(t *testing.T) {
// 	ctx := createContext(t, time.Second*common.DefaultTimeout)
// 	// connect
// 	mc := createMilvusClient(ctx, t)

// 	// create collection
// 	cp := CollectionParams{CollectionFieldsType: Int64FloatVec, AutoID: false, EnableDynamicField: true,
// 		ShardsNum: common.DefaultShards, Dim: common.DefaultDim}
// 	collName := createCollection(ctx, t, mc, cp)

// 	// insert
// 	dpColumns := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
// 		start: 0, nb: common.DefaultNb, dim: common.DefaultDim, EnableDynamicField: true, WithRows: false}
// 	_, _ = insertData(ctx, t, mc, dpColumns)

// 	mc.Flush(ctx, collName, false)

// 	/* dpRows := DataParams{CollectionName: collName, PartitionName: "", CollectionFieldsType: Int64FloatVec,
// 	   start: common.DefaultNb, nb: common.DefaultNb * 2, dim: common.DefaultDim, EnableDynamicField: true, WithRows: true}
// 	_, _ = insertData(ctx, t, mc, dpRows)*/

// 	idx, _ := entity.NewIndexHNSW(entity.COSINE, 8, 96)
// 	_ = mc.CreateIndex(ctx, collName, common.DefaultFloatVecFieldName, idx, false)

// 	// Load collection
// 	errLoad := mc.LoadCollection(ctx, collName, false)
// 	common.CheckErr(t, errLoad, true)

// 	// search iterator with default batch
// 	sp, _ := entity.NewIndexHNSWSearchParam(80)
// 	queryVec := common.GenSearchVectors(1, common.DefaultDim, entity.FieldTypeFloatVector)
// 	opt := client.NewSearchIteratorOption(collName, common.DefaultFloatVecFieldName, sp, queryVec[0], entity.COSINE)
// 	itr, err := mc.SearchIterator(ctx, opt)
// 	common.CheckErr(t, err, true)
// 	// common.CheckSearchIteratorResult(ctx, t, itr, common.DefaultNb)
// 	var cnt int
// 	for {
// 		sr, err := itr.Next(ctx)
// 		if err != nil {
// 			if err == io.EOF {
// 				break
// 			}
// 			t.FailNow()
// 		}
// 		cnt += sr.IDs.Len()
// 		time.Sleep(time.Second)
// 	}
// 	log.Println(cnt)
// }
