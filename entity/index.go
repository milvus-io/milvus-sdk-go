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

// Package entity defines entities used in sdk
package entity

import "github.com/milvus-io/milvus-sdk-go/internal/proto/common"

// IndexState export index state
type IndexState common.IndexState

// IndexType index type
type IndexType string

// MetricType metric type
type MetricType string

// Index Constants
const (
	Flat                 IndexType = "FLAT"     //faiss
	IvfFlat              IndexType = "IVF_FLAT" //faiss
	IvfPQ                IndexType = "IVF_PQ"   //faiss
	IndexFaissIvfSQ8     IndexType = "IVF_SQ8"
	IndexFaissIvfSQ8H    IndexType = "IVF_SQ8_HYBRID"
	IndexFaissBinIDMap   IndexType = "BIN_FLAT"
	IndexFaissBinIvfFlat IndexType = "BIN_IVF_FLAT"
	IndexNSG             IndexType = "NSG"
	IndexHNSW            IndexType = "HNSW"
	IndexRHNSWFlat       IndexType = "RHNSW_FLAT"
	IndexRHNSWPQ         IndexType = "RHNSW_PQ"
	IndexRHNSWSQ         IndexType = "RHNSW_SQ"
	IndexANNOY           IndexType = "ANNOY"
	IndexNGTPANNG        IndexType = "NGT_PANNG"
	IndexNGTONNG         IndexType = "NGT_ONNG"
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

func NewFlatIndex(m MetricType) Index {
	return flatIndex{
		baseIndex: baseIndex{it: Flat},
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