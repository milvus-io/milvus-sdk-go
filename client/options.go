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

package client

import (
	"fmt"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// CreateCollectionOption is an option that is used to alter create collection options.
type CreateCollectionOption func(opt *createCollOpt)

type createCollOpt struct {
	ConsistencyLevel    entity.ConsistencyLevel
	NumPartitions       int64
	PrimaryKeyFieldName string
	PrimaryKeyFieldType entity.FieldType
	PrimaryKeyMaxLength int64
	VectorFieldName     string
	MetricsType         entity.MetricType
	AutoID              bool
	EnableDynamicSchema bool
	Properties          map[string]string
	MsgBase             *commonpb.MsgBase
}

func WithPKFieldName(name string) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.PrimaryKeyFieldName = name
	}
}

func WithPKFieldType(tp entity.FieldType) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.PrimaryKeyFieldType = tp
	}
}

func WithPKMaxLength(maxLength int64) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.PrimaryKeyMaxLength = maxLength
	}
}

func WithVectorFieldName(name string) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.VectorFieldName = name
	}
}

func WithMetricsType(mt entity.MetricType) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.MetricsType = mt
	}
}

func WithAutoID(autoID bool) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.AutoID = autoID
	}
}

func WithEnableDynamicSchema(enable bool) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.EnableDynamicSchema = enable
	}
}

// WithConsistencyLevel specifies a specific ConsistencyLevel, rather than using the default ReaderProperties.
func WithConsistencyLevel(cl entity.ConsistencyLevel) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.ConsistencyLevel = cl
	}
}

// WithPartitionNum returns a create collection options to set physical partition number when logical partition feature.
func WithPartitionNum(partitionNums int64) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.NumPartitions = partitionNums
	}
}

func WithCollectionProperty(key, value string) CreateCollectionOption {
	return func(opt *createCollOpt) {
		if opt.Properties == nil {
			opt.Properties = make(map[string]string)
		}
		opt.Properties[key] = value
	}
}

// LoadCollectionOption is an option that is used to modify LoadCollectionRequest
type LoadCollectionOption func(*milvuspb.LoadCollectionRequest)

// WithReplicaNumber specifies a specific ReplicaNumber, rather than using the default ReplicaNumber.
func WithReplicaNumber(rn int32) LoadCollectionOption {
	return func(req *milvuspb.LoadCollectionRequest) {
		req.ReplicaNumber = rn
	}
}

// WithResourceGroups specifies some specific ResourceGroup(s) to load the replica(s), rather than using the default ResourceGroup.
func WithResourceGroups(rgs []string) LoadCollectionOption {
	return func(req *milvuspb.LoadCollectionRequest) {
		req.ResourceGroups = rgs
	}
}

// SearchQueryOption is an option of search/query request
type SearchQueryOption struct {
	// Consistency Level
	ConsistencyLevel   entity.ConsistencyLevel
	GuaranteeTimestamp uint64
	// Pagination
	Limit  int64
	Offset int64

	IgnoreGrowing bool
	ForTuning     bool

	GroupByField string
}

// SearchQueryOptionFunc is a function which modifies SearchOption
type SearchQueryOptionFunc func(option *SearchQueryOption)

func WithForTuning() SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.ForTuning = true
	}
}

func WithIgnoreGrowing() SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.IgnoreGrowing = true
	}
}

// WithOffset returns search/query option with offset.
func WithOffset(offset int64) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.Offset = offset
	}
}

// WithLimit returns search/query option with limit.
func WithLimit(limit int64) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.Limit = limit
	}
}

func WithGroupByField(groupByField string) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.GroupByField = groupByField
	}
}

// WithSearchQueryConsistencyLevel specifies consistency level
func WithSearchQueryConsistencyLevel(cl entity.ConsistencyLevel) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.ConsistencyLevel = cl
	}
}

// WithGuaranteeTimestamp specifies guarantee timestamp
func WithGuaranteeTimestamp(gt uint64) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.GuaranteeTimestamp = gt
	}
}

// Deprecated: time travel is not supported since v2.3.0
func WithTravelTimestamp(_ uint64) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {}
}

func makeSearchQueryOption(collName string, opts ...SearchQueryOptionFunc) (*SearchQueryOption, error) {
	opt := &SearchQueryOption{
		ConsistencyLevel: entity.ClBounded, // default
	}
	info, ok := MetaCache.getCollectionInfo(collName)
	if ok {
		opt.ConsistencyLevel = info.ConsistencyLevel
	}
	for _, o := range opts {
		o(opt)
	}
	// sanity-check
	if opt.ConsistencyLevel != entity.ClCustomized && opt.GuaranteeTimestamp != 0 {
		return nil, errors.New("user can only specify guarantee timestamp under customized consistency level")
	}

	switch opt.ConsistencyLevel {
	case entity.ClStrong:
		opt.GuaranteeTimestamp = StrongTimestamp
	case entity.ClSession:
		ts, ok := MetaCache.getSessionTs(collName)
		if !ok {
			ts = EventuallyTimestamp
		}
		opt.GuaranteeTimestamp = ts
	case entity.ClBounded:
		opt.GuaranteeTimestamp = BoundedTimestamp
	case entity.ClEventually:
		opt.GuaranteeTimestamp = EventuallyTimestamp
	case entity.ClCustomized:
		// respect opt.GuaranteeTimestamp
	}
	return opt, nil
}

// BulkInsertOption is an option that is used to modify ImportRequest
type BulkInsertOption func(request *milvuspb.ImportRequest)

// WithStartTs specifies a specific startTs
func WithStartTs(startTs int64) BulkInsertOption {
	return func(req *milvuspb.ImportRequest) {
		optionMap := entity.KvPairsMap(req.GetOptions())
		optionMap["start_ts"] = fmt.Sprint(startTs)
		req.Options = entity.MapKvPairs(optionMap)
	}
}

// WithEndTs specifies a specific endTs
func WithEndTs(endTs int64) BulkInsertOption {
	return func(req *milvuspb.ImportRequest) {
		optionMap := entity.KvPairsMap(req.GetOptions())
		optionMap["end_ts"] = fmt.Sprint(endTs)
		req.Options = entity.MapKvPairs(optionMap)
	}
}

// IsBackup specifies it is triggered by backup tool
func IsBackup() BulkInsertOption {
	return func(req *milvuspb.ImportRequest) {
		optionMap := entity.KvPairsMap(req.GetOptions())
		optionMap["backup"] = "true"
		req.Options = entity.MapKvPairs(optionMap)
	}
}

type getOption struct {
	partitionNames []string
	outputFields   []string
}

type GetOption func(o *getOption)

func GetWithPartitions(partionNames ...string) GetOption {
	return func(o *getOption) {
		o.partitionNames = partionNames
	}
}

func GetWithOutputFields(outputFields ...string) GetOption {
	return func(o *getOption) {
		o.outputFields = outputFields
	}
}

type DropCollectionOption func(*milvuspb.DropCollectionRequest)

type ReleaseCollectionOption func(*milvuspb.ReleaseCollectionRequest)

type FlushOption func(*milvuspb.FlushRequest)

type CreateDatabaseOption func(*milvuspb.CreateDatabaseRequest)

type DropDatabaseOption func(*milvuspb.DropDatabaseRequest)

type ReplicateMessageOption func(*milvuspb.ReplicateMessageRequest)

type CreatePartitionOption func(*milvuspb.CreatePartitionRequest)

type DropPartitionOption func(*milvuspb.DropPartitionRequest)

type LoadPartitionsOption func(*milvuspb.LoadPartitionsRequest)

type ReleasePartitionsOption func(*milvuspb.ReleasePartitionsRequest)
