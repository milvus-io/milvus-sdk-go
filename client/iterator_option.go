package client

import "github.com/milvus-io/milvus-sdk-go/v2/entity"

func NewSearchIteratorOption(collectionName string, vectorFieldName string, sp entity.SearchParam, vector entity.Vector, metricType entity.MetricType) *SearchIteratorOption {
	return &SearchIteratorOption{
		collectionName: collectionName,
		vectorField:    vectorFieldName,
		batchSize:      1000,
		searchParam:    sp,
		vector:         vector,
		metricType:     metricType,
	}
}

type SearchIteratorOption struct {
	collectionName string
	partitionNames []string
	expr           string
	outputFields   []string
	batchSize      int

	vectorField string
	searchParam entity.SearchParam
	vector      entity.Vector
	metricType  entity.MetricType
}

func (opt *SearchIteratorOption) WithPartitions(partitionNames ...string) *SearchIteratorOption {
	opt.partitionNames = partitionNames
	return opt
}

func (opt *SearchIteratorOption) WithExpr(expr string) *SearchIteratorOption {
	opt.expr = expr
	return opt
}

func (opt *SearchIteratorOption) WithOutputFields(outputFields ...string) *SearchIteratorOption {
	opt.outputFields = outputFields
	return opt
}

func (opt *SearchIteratorOption) WithBatchSize(batchSize int) *SearchIteratorOption {
	opt.batchSize = batchSize
	return opt
}

func NewQueryIteratorOption(collectionName string) *QueryIteratorOption {
	return &QueryIteratorOption{
		collectionName: collectionName,
		batchSize:      1000,
	}
}

type QueryIteratorOption struct {
	collectionName string
	partitionNames []string
	expr           string
	outputFields   []string
	batchSize      int
}

func (opt *QueryIteratorOption) WithPartitions(partitionNames ...string) *QueryIteratorOption {
	opt.partitionNames = partitionNames
	return opt
}

func (opt *QueryIteratorOption) WithExpr(expr string) *QueryIteratorOption {
	opt.expr = expr
	return opt
}

func (opt *QueryIteratorOption) WithOutputFields(outputFields ...string) *QueryIteratorOption {
	opt.outputFields = outputFields
	return opt
}

func (opt *QueryIteratorOption) WithBatchSize(batchSize int) *QueryIteratorOption {
	opt.batchSize = batchSize
	return opt
}
