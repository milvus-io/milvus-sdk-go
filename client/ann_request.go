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

func NewANNSearchRequest(fieldName string, metricsType entity.MetricType, vectors []entity.Vector, searchParam entity.SearchParam, limit int, options ...SearchQueryOptionFunc) *ANNSearchRequest {
	return &ANNSearchRequest{
		fieldName:   fieldName,
		vectors:     vectors,
		metricType:  metricsType,
		searchParam: searchParam,
		limit:       limit,
	}
}
func (r *ANNSearchRequest) WithExpr(expr string) *ANNSearchRequest {
	r.expr = expr
	return r
}

func (req *ANNSearchRequest) getMilvusSearchRequest(collectionInfo *collInfo) (*milvuspb.SearchRequest, error) {
	opt := &SearchQueryOption{
		ConsistencyLevel: collectionInfo.ConsistencyLevel, // default
	}
	for _, o := range req.options {
		o(opt)
	}
	params := req.searchParam.Params()
	params[forTuningKey] = opt.ForTuning
	bs, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":     req.fieldName,
		"topk":           fmt.Sprintf("%d", req.limit),
		"params":         string(bs),
		"metric_type":    string(req.metricType),
		"round_decimal":  "-1",
		ignoreGrowingKey: strconv.FormatBool(opt.IgnoreGrowing),
		offsetKey:        fmt.Sprintf("%d", opt.Offset),
		groupByKey:       opt.GroupByField,
	})

	result := &milvuspb.SearchRequest{
		DbName:             "",
		Dsl:                req.expr,
		PlaceholderGroup:   vector2PlaceholderGroupBytes(req.vectors),
		DslType:            commonpb.DslType_BoolExprV1,
		SearchParams:       searchParams,
		GuaranteeTimestamp: opt.GuaranteeTimestamp,
		Nq:                 int64(len(req.vectors)),
	}
	return result, nil
}
