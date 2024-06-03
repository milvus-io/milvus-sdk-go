package client

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

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

func (c *GrpcClient) QueryIterator(ctx context.Context, opt *QueryIteratorOption) (*QueryIterator, error) {
	collectionName := opt.collectionName
	var sch *entity.Schema
	collInfo, ok := MetaCache.getCollectionInfo(collectionName)
	if !ok {
		coll, err := c.DescribeCollection(ctx, collectionName)
		if err != nil {
			return nil, err
		}
		sch = coll.Schema
	} else {
		sch = collInfo.Schema
	}

	itr := &QueryIterator{
		client: c,

		collectionName: opt.collectionName,
		partitionNames: opt.partitionNames,
		outputFields:   opt.outputFields,
		sch:            sch,
		pkField:        sch.PKField(),

		batchSize: opt.batchSize,
		expr:      opt.expr,
	}

	err := itr.init(ctx)
	if err != nil {
		return nil, err
	}
	return itr, nil
}

type QueryIterator struct {
	// user provided expression
	expr string

	batchSize int

	cached ResultSet

	collectionName string
	partitionNames []string
	outputFields   []string
	sch            *entity.Schema
	pkField        *entity.Field

	lastPK interface{}

	// internal grpc client
	client *GrpcClient
}

// init fetches the first batch of data and put it into cache.
// this operation could be used to check all the parameters before returning the iterator.
func (itr *QueryIterator) init(ctx context.Context) error {
	rs, err := itr.fetchNextBatch(ctx)
	if err != nil {
		return err
	}
	itr.cached = rs
	return nil
}

func (itr *QueryIterator) composeIteratorExpr() string {
	if itr.lastPK == nil {
		return itr.expr
	}

	expr := strings.TrimSpace(itr.expr)

	switch itr.pkField.DataType {
	case entity.FieldTypeInt64:
		expr = fmt.Sprintf("(%s) and %s > %d", expr, itr.pkField.Name, itr.lastPK)
	case entity.FieldTypeVarChar:
		expr += fmt.Sprintf(`(%s) and %s > "%s"`, expr, itr.pkField.Name, itr.lastPK)
	default:
		return itr.expr
	}
	return expr
}

func (itr *QueryIterator) fetchNextBatch(ctx context.Context) (ResultSet, error) {
	return itr.client.Query(ctx, itr.collectionName, itr.partitionNames, itr.composeIteratorExpr(), itr.outputFields,
		WithLimit(int64(float64(itr.batchSize))), withIterator(), reduceForBest(true))
}

func (itr *QueryIterator) cachedSufficient() bool {
	return itr.cached != nil && itr.cached.Len() >= itr.batchSize
}

func (itr *QueryIterator) cacheNextBatch(rs ResultSet) (ResultSet, error) {
	result := rs.Slice(0, itr.batchSize)
	itr.cached = rs.Slice(itr.batchSize, -1)

	pkColumn := result.GetColumn(itr.pkField.Name)
	switch itr.pkField.DataType {
	case entity.FieldTypeInt64:
		itr.lastPK, _ = pkColumn.GetAsInt64(pkColumn.Len() - 1)
	case entity.FieldTypeVarChar:
		itr.lastPK, _ = pkColumn.GetAsString(pkColumn.Len() - 1)
	default:
		return nil, errors.Newf("unsupported pk type: %v", itr.pkField.DataType)
	}
	return result, nil
}

func (itr *QueryIterator) Next(ctx context.Context) (ResultSet, error) {
	var rs ResultSet
	var err error

	// check cache sufficient for next batch
	if !itr.cachedSufficient() {
		rs, err = itr.fetchNextBatch(ctx)
		if err != nil {
			return nil, err
		}
	} else {
		rs = itr.cached
	}

	// if resultset is empty, return EOF
	if rs.Len() == 0 {
		return nil, io.EOF
	}

	return itr.cacheNextBatch(rs)
}
