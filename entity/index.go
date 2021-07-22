// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

package entity

import "github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"

//go:generate go run genidx/genidx.go

// IndexState export index state
type IndexState common.IndexState

// IndexType index type
type IndexType string

// MetricType metric type
type MetricType string

// Index Constants
const (
	Flat       IndexType = "FLAT" //faiss
	BinFlat    IndexType = "BIN_FLAT"
	IvfFlat    IndexType = "IVF_FLAT" //faiss
	BinIvfFlat IndexType = "BIN_IVF_FLAT"
	IvfPQ      IndexType = "IVF_PQ" //faiss
	IvfSQ8     IndexType = "IVF_SQ8"
	IvfSQ8H    IndexType = "IVF_SQ8_HYBRID"
	NSG        IndexType = "NSG"
	HNSW       IndexType = "HNSW"
	RHNSWFlat  IndexType = "RHNSW_FLAT"
	RHNSWPQ    IndexType = "RHNSW_PQ"
	RHNSWSQ    IndexType = "RHNSW_SQ"
	IvfHNSW    IndexType = "IVF_HNSW"
	ANNOY      IndexType = "ANNOY"
	NGTPANNG   IndexType = "NGT_PANNG"
	NGTONNG    IndexType = "NGT_ONNG"
)

// Metric Constants
const (
	L2             MetricType = "L2"
	IP             MetricType = "IP"
	HAMMING        MetricType = "HAMMING"
	JACCARD        MetricType = "JACCARD"
	TANIMOTO       MetricType = "TANIMOTO"
	SUBSTRUCTURE   MetricType = "SUBSTRUCTURE"
	SUPERSTRUCTURE MetricType = "SUPERSTRUCTURE"
)

// index param field tag
const (
	tIndexType  = `index_type`
	tMetricType = `metric_type`
)

// Index represent index in milvus
type Index interface {
	Name() string
	IndexType() IndexType
	Params() map[string]string
}

// SearchParam interface for index related search param
type SearchParam interface {
	// returns parameters for search/query
	Params() map[string]interface{}
}

type baseIndex struct {
	it   IndexType
	name string
}

// Name implements Index
func (b baseIndex) Name() string {
	return b.name
}

// IndexType implements Index
func (b baseIndex) IndexType() IndexType {
	return b.it
}

type flatIndex struct {
	baseIndex
	m MetricType
}

func (f flatIndex) Params() map[string]string {
	return map[string]string{
		tIndexType:  string(Flat),
		tMetricType: string(f.m),
	}
}

func NewFlatIndex(name string, m MetricType) Index {
	return flatIndex{
		baseIndex: baseIndex{it: Flat, name: name},
		m:         m,
	}
}

// GenericIndex index struct for general usage
// no constraint for index is applied
type GenericIndex struct {
	baseIndex
	params map[string]string
}

// Params implements Index
func (gi GenericIndex) Params() map[string]string {
	m := map[string]string{
		tIndexType: string(gi.IndexType()),
	}
	for k, v := range gi.params {
		m[k] = v
	}
	return m
}

// NewGenericIndex create generic index instance
func NewGenericIndex(name string, it IndexType, params map[string]string) Index {
	return GenericIndex{
		baseIndex: baseIndex{
			it:   it,
			name: name,
		},
		params: params,
	}
}
