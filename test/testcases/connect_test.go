//go:build L0

package testcases

import (
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/test/base"
	"github.com/milvus-io/milvus-sdk-go/v2/test/common"
)

// test connect and close, connect again
func TestConnectClose(t *testing.T) {
	// connect
	ctx := createContext(t, time.Second*common.DefaultTimeout)
	mc, errConnect := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect, true)

	// connect success
	_, errList := mc.ListCollections(ctx)
	common.CheckErr(t, errList, true)

	// close success
	mc.Close()
	_, errList2 := mc.ListCollections(ctx)
	common.CheckErr(t, errList2, false, "the client connection is closing")

	// connect again
	mc, errConnect2 := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect2, true)
	_, errList3 := mc.ListCollections(ctx)
	common.CheckErr(t, errList3, true)
}

// test connect with timeout and invalid addr
func TestConnectInvalidAddr(t *testing.T) {
	t.Skipf("Issue: %s", "https://github.com/milvus-io/milvus-sdk-go/issues/345")
	// connect
	ctx := createContext(t, time.Second*10)

	_, errConnect := base.NewDefaultMilvusClient(ctx, "aa")
	common.CheckErr(t, errConnect, false, "xxx")
}

// test connect repeatedly
func TestConnectRepeat(t *testing.T) {
	// connect
	ctx := createContext(t, time.Second*10)

	_, errConnect := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect, true)

	// connect again
	mc, errConnect2 := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect2, true)

	_, err := mc.ListCollections(ctx)
	common.CheckErr(t, err, true)
}

// test close repeatedly
func TestCloseRepeat(t *testing.T) {
	// connect
	ctx := createContext(t, time.Second*10)
	mc, errConnect2 := base.NewDefaultMilvusClient(ctx, *addr)
	common.CheckErr(t, errConnect2, true)

	// close and again
	mc.Close()
	mc.Close()
}
