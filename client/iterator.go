package client

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func (c *GrpcClient) SearchIterator(ctx context.Context, opt *SearchIteratorOption) (*SearchIterator, error) {
	collectionName := opt.collectionName
	collInfo, err := c.getCollectionInfo(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	sch := collInfo.Schema

	var vectorField *entity.Field
	for _, field := range sch.Fields {
		if field.Name == opt.vectorField {
			vectorField = field
		}
	}
	if vectorField == nil {
		return nil, errors.Newf("vector field %s not found", opt.vectorField)
	}

	itr := &SearchIterator{
		client: c,

		collectionName: opt.collectionName,
		partitionNames: opt.partitionNames,
		outputFields:   opt.outputFields,
		sch:            sch,
		pkField:        sch.PKField(),
		vectorField:    vectorField,

		searchParam: opt.searchParam,
		vector:      opt.vector,
		metricType:  opt.metricType,

		batchSize: opt.batchSize,
		expr:      opt.expr,
	}

	err = itr.init(ctx)
	if err != nil {
		return nil, err
	}
	return itr, nil
}

type SearchIterator struct {
	// user provided expression
	expr string

	batchSize int

	cached *SearchResult

	collectionName string
	partitionNames []string
	outputFields   []string
	sch            *entity.Schema
	pkField        *entity.Field
	vectorField    *entity.Field

	searchParam entity.SearchParam
	vector      entity.Vector
	metricType  entity.MetricType

	lastPKs      []interface{}
	lastDistance float32

	// internal grpc client
	client *GrpcClient
}

func (itr SearchIterator) init(ctx context.Context) error {
	if itr.batchSize <= 0 {
		return errors.New("batch size cannot less than 1")
	}

	rs, err := itr.fetchNextBatch(ctx)
	if err != nil {
		return err
	}
	itr.cached = rs
	return nil
}

func (itr *SearchIterator) composeIteratorExpr() string {
	if len(itr.lastPKs) == 0 {
		return itr.expr
	}

	expr := strings.TrimSpace(itr.expr)

	switch itr.pkField.DataType {
	case entity.FieldTypeInt64:
		values := make([]string, 0, len(itr.lastPKs))
		for _, pk := range itr.lastPKs {
			values = append(values, fmt.Sprintf("%v", pk))
		}
		if len(expr) == 0 {
			expr = fmt.Sprintf("%s not in [%s]", itr.pkField.Name, strings.Join(values, ","))
		} else {
			expr = fmt.Sprintf("(%s) and %s not in [%s]", expr, itr.pkField.Name, strings.Join(values, ","))
		}
	case entity.FieldTypeVarChar:
		values := make([]string, 0, len(itr.lastPKs))
		for _, pk := range itr.lastPKs {
			values = append(values, fmt.Sprintf("\"%v\"", pk))
		}
		if len(expr) == 0 {
			expr = fmt.Sprintf("%s not in [%s]", itr.pkField.Name, strings.Join(values, ","))
		} else {
			expr = fmt.Sprintf("(%s) and %s not in [%s]", expr, itr.pkField.Name, strings.Join(values, ","))
		}
	default:
		return itr.expr
	}
	return expr
}

func (itr *SearchIterator) fetchNextBatch(ctx context.Context) (*SearchResult, error) {
	rss, err := itr.client.Search(ctx, itr.collectionName, itr.partitionNames, itr.composeIteratorExpr(), itr.outputFields, []entity.Vector{itr.vector}, itr.vectorField.Name, itr.metricType, itr.batchSize, itr.searchParam)
	if err != nil {
		return nil, err
	}
	return &rss[0], nil
}

func (itr *SearchIterator) cachedSufficient() bool {
	return itr.cached != nil && itr.cached.IDs.Len() >= itr.batchSize
}

func (itr *SearchIterator) cacheNextBatch(rs *SearchResult) (*SearchResult, error) {
	result := rs.Slice(0, itr.batchSize)
	itr.cached = rs.Slice(itr.batchSize, -1)

	if result.IDs.Len() == 0 {
		return nil, io.EOF
	}
	nextRangeFilter := result.Scores[len(result.Scores)-1]
	// setup last score
	itr.lastDistance = nextRangeFilter
	if nextRangeFilter != itr.lastDistance {
		itr.lastPKs = nil
	}
	pk, err := result.IDs.Get(len(result.Scores) - 1)
	if err != nil {
		return nil, err
	}
	itr.lastPKs = append(itr.lastPKs, pk)
	for i := len(result.Scores) - 2; i >= 0; i-- {
		if result.Scores[i] != nextRangeFilter {
			break
		}
		pk, err := result.IDs.Get(i)
		if err != nil {
			return nil, err
		}
		itr.lastPKs = append(itr.lastPKs, pk)
	}

	itr.searchParam.AddRangeFilter(float64(itr.lastDistance))

	return result, nil
}

func (itr *SearchIterator) Next(ctx context.Context) (*SearchResult, error) {
	var rs *SearchResult
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
	if rs.IDs.Len() == 0 {
		return nil, io.EOF
	}

	return itr.cacheNextBatch(rs)
}

func (c *GrpcClient) QueryIterator(ctx context.Context, opt *QueryIteratorOption) (*QueryIterator, error) {
	collectionName := opt.collectionName
	collInfo, err := c.getCollectionInfo(ctx, collectionName)
	if err != nil {
		return nil, err
	}
	sch := collInfo.Schema

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

	err = itr.init(ctx)
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
	if itr.batchSize <= 0 {
		return errors.New("batch size cannot less than 1")
	}

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
		if len(expr) == 0 {
			expr = fmt.Sprintf("%s > %d", itr.pkField.Name, itr.lastPK)
		} else {
			expr = fmt.Sprintf("(%s) and %s > %d", expr, itr.pkField.Name, itr.lastPK)
		}
	case entity.FieldTypeVarChar:
		if len(expr) == 0 {
			expr = fmt.Sprintf(`%s > "%s"`, itr.pkField.Name, itr.lastPK)
		} else {
			expr = fmt.Sprintf(`(%s) and %s > "%s"`, expr, itr.pkField.Name, itr.lastPK)
		}
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
