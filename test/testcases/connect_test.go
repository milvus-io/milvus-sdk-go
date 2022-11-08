package testcases

import (
	"context"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

func TestConnectAndClose(t *testing.T) {
	// connect
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*common.DefaultTimeout)
	defer cancel()
	mc, errConnect := base.NewMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect, true, "")

	// connect success
	_, errList := mc.ListCollections(ctx)
	common.CheckErr(t, errList, true, "")

	// close success
	mc.Close()
	_, errList2 := mc.ListCollections(ctx)
	common.CheckErr(t, errList2, false, "the client connection is closing")
}
