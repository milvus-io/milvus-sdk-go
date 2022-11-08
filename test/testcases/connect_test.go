package testcases

import (
	"context"
	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
	"testing"
)

func TestConnectAndClose(t *testing.T)  {
	// connect
	mc, errConnect := base.NewMilvusClient(context.Background(), *addr)
	common.CheckErr(t, errConnect, true, "")

	// connect success
	_, errList := mc.ListCollections(context.Background())
	common.CheckErr(t, errList, true, "")

	// close success
	mc.Close()
	_, errList2 := mc.ListCollections(context.Background())
	common.CheckErr(t, errList2, false, "the client connection is closing")
}


