package entity

import (
	"encoding/json"
	"fmt"

	"github.com/cockroachdb/errors"
)

var _ Index = (*IndexSparseInverted)(nil)

// IndexSparseInverted index type for SPARSE_INVERTED_INDEX
type IndexSparseInverted struct {
	metricType MetricType
	dropRatio  float64
}

func (i *IndexSparseInverted) Name() string {
	return "SparseInverted"
}

func (i *IndexSparseInverted) IndexType() IndexType {
	return SparseInverted
}

func (i *IndexSparseInverted) Params() map[string]string {
	params := map[string]string{
		"drop_ratio_build": fmt.Sprintf("%v", i.dropRatio),
	}
	bs, _ := json.Marshal(params)
	return map[string]string{
		tParams:     string(bs),
		tIndexType:  string(i.IndexType()),
		tMetricType: string(i.metricType),
	}
}

type IndexSparseInvertedSearchParam struct {
	baseSearchParams
}

func NewIndexSparseInvertedSearchParam(dropRatio float64) (*IndexSparseInvertedSearchParam, error) {
	if dropRatio < 0 || dropRatio >= 1 {
		return nil, errors.Newf("invalid dropRatio for search: %v, must be in range [0, 1)", dropRatio)
	}
	sp := &IndexSparseInvertedSearchParam{
		baseSearchParams: newBaseSearchParams(),
	}

	sp.params["drop_ratio_search"] = dropRatio
	return sp, nil
}

// IndexSparseInverted index type for SPARSE_INVERTED_INDEX
func NewIndexSparseInverted(metricType MetricType, dropRatio float64) (*IndexSparseInverted, error) {
	if dropRatio < 0 || dropRatio >= 1.0 {
		return nil, errors.Newf("invalid dropRatio for build: %v, must be in range [0, 1)", dropRatio)
	}
	return &IndexSparseInverted{
		metricType: metricType,
		dropRatio:  dropRatio,
	}, nil
}

type IndexSparseWAND struct {
	metricType MetricType
	dropRatio  float64
}

func (i *IndexSparseWAND) Name() string {
	return "SparseWAND"
}

func (i *IndexSparseWAND) IndexType() IndexType {
	return SparseWAND
}

func (i *IndexSparseWAND) Params() map[string]string {
	params := map[string]string{
		"drop_ratio_build": fmt.Sprintf("%v", i.dropRatio),
	}
	bs, _ := json.Marshal(params)
	return map[string]string{
		tParams:     string(bs),
		tIndexType:  string(i.IndexType()),
		tMetricType: string(i.metricType),
	}
}

// IndexSparseWAND index type for SPARSE_WAND, weak-and
func NewIndexSparseWAND(metricType MetricType, dropRatio float64) (*IndexSparseWAND, error) {
	if dropRatio < 0 || dropRatio >= 1.0 {
		return nil, errors.Newf("invalid dropRatio for build: %v, must be in range [0, 1)", dropRatio)
	}
	return &IndexSparseWAND{
		metricType: metricType,
		dropRatio:  dropRatio,
	}, nil
}

type IndexSparseWANDSearchParam struct {
	baseSearchParams
}

func NewIndexSparseWANDSearchParam(dropRatio float64) (*IndexSparseWANDSearchParam, error) {
	if dropRatio < 0 || dropRatio >= 1 {
		return nil, errors.Newf("invalid dropRatio for search: %v, must be in range [0, 1)", dropRatio)
	}
	sp := &IndexSparseWANDSearchParam{
		baseSearchParams: newBaseSearchParams(),
	}

	sp.params["drop_ratio_search"] = dropRatio
	return sp, nil
}
