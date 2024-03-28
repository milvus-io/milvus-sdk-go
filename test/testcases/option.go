package testcases

import (
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type HelpPartitionColumns struct {
	PartitionName string
	IdsColumn     entity.Column
	VectorColumn  entity.Column
}

type CollectionFieldsType string

type CollectionParams struct {
	CollectionFieldsType CollectionFieldsType // collection fields type
	AutoID               bool                 // autoId
	EnableDynamicField   bool                 // enable dynamic field
	ShardsNum            int32
	Dim                  int64
	MaxLength            int64
	MaxCapacity          int64
}

type DataParams struct {
	CollectionName       string // insert data into which collection
	PartitionName        string
	CollectionFieldsType CollectionFieldsType // collection fields type
	start                int                  // start
	nb                   int                  // insert how many data
	dim                  int64
	EnableDynamicField   bool // whether insert dynamic field data
	WithRows             bool
	DoInsert             bool
}

func (d DataParams) IsEmpty() bool {
	return d.CollectionName == "" || d.nb == 0
}

type FlushParams struct {
	DoFlush        bool
	PartitionNames []string
	async          bool
}

type IndexParams struct {
	BuildIndex bool
	Index      entity.Index
	FieldName  string
	async      bool
}

func (i IndexParams) IsEmpty() bool {
	return i.Index == nil || i.FieldName == ""
}

type LoadParams struct {
	DoLoad         bool
	PartitionNames []string
	async          bool
}

type ClientParamsOption struct {
	DataParams  DataParams
	FlushParams FlushParams
	IndexParams []IndexParams
	LoadParams  LoadParams
	CreateOpts  client.CreateCollectionOption
	IndexOpts   client.IndexOption
	LoadOpts    client.LoadCollectionOption
}

type PrepareCollectionOption func(opt *ClientParamsOption)

func WithDataParams(dp DataParams) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.DataParams = dp
	}
}

func WithFlushParams(fp FlushParams) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.FlushParams = fp
	}
}

func WithIndexParams(ips []IndexParams) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.IndexParams = ips
	}
}

func WithLoadParams(lp LoadParams) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.LoadParams = lp
	}
}

func WithCreateOption(createOpts client.CreateCollectionOption) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.CreateOpts = createOpts
	}
}

func WithIndexOption(indexOpts client.IndexOption) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.IndexOpts = indexOpts
	}
}

func WithLoadOption(loadOpts client.LoadCollectionOption) PrepareCollectionOption {
	return func(opt *ClientParamsOption) {
		opt.LoadOpts = loadOpts
	}
}
