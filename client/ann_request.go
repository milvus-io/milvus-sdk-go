package client

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type ANNSearchRequest struct {
	fieldName   string
	vectors     []entity.Vector
	metricType  entity.MetricType
	expr        string
	searchParam entity.SearchParam
	options     []SearchQueryOptionFunc
	limit       int
}

func NewANNSearchRequest(fieldName string, metricsType entity.MetricType, expr string, vectors []entity.Vector, searchParam entity.SearchParam, limit int, options ...SearchQueryOptionFunc) *ANNSearchRequest {
	return &ANNSearchRequest{
		fieldName:   fieldName,
		vectors:     vectors,
		metricType:  metricsType,
		expr:        expr,
		searchParam: searchParam,
		limit:       limit,
		options:     options,
	}
}
func (r *ANNSearchRequest) WithExpr(expr string) *ANNSearchRequest {
	r.expr = expr
	return r
}

func (r *ANNSearchRequest) getMilvusSearchRequest(collectionInfo *collInfo, opts ...SearchQueryOptionFunc) (*milvuspb.SearchRequest, error) {
	opt := &SearchQueryOption{
		ConsistencyLevel: collectionInfo.ConsistencyLevel, // default
	}
	for _, o := range r.options {
		o(opt)
	}
	for _, o := range opts {
		o(opt)
	}
	params := r.searchParam.Params()
	params[forTuningKey] = opt.ForTuning
	bs, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":     r.fieldName,
		"topk":           fmt.Sprintf("%d", r.limit),
		"params":         string(bs),
		"metric_type":    string(r.metricType),
		"round_decimal":  "-1",
		ignoreGrowingKey: strconv.FormatBool(opt.IgnoreGrowing),
		offsetKey:        fmt.Sprintf("%d", opt.Offset),
		groupByKey:       opt.GroupByField,
	})

	result := &milvuspb.SearchRequest{
		DbName:             "",
		Dsl:                r.expr,
		PlaceholderGroup:   vector2PlaceholderGroupBytes(r.vectors),
		DslType:            commonpb.DslType_BoolExprV1,
		SearchParams:       searchParams,
		GuaranteeTimestamp: opt.GuaranteeTimestamp,
		Nq:                 int64(len(r.vectors)),
	}
	return result, nil
}
