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
	"strconv"

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

func WithLoadFields(fields ...string) LoadCollectionOption {
	return func(req *milvuspb.LoadCollectionRequest) {
		req.LoadFields = fields
	}
}

func WithSkipDynamicFields(skip bool) LoadCollectionOption {
	return func(req *milvuspb.LoadCollectionRequest) {
		req.SkipLoadDynamicField = skip
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

	isIterator    bool
	reduceForBest bool
}

// SearchQueryOptionFunc is a function which modifies SearchOption
type SearchQueryOptionFunc func(option *SearchQueryOption)

func withIterator() SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.isIterator = true
	}
}

func reduceForBest(value bool) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.reduceForBest = value
	}
}

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

func (c *GrpcClient) makeSearchQueryOption(collName string, opts ...SearchQueryOptionFunc) (*SearchQueryOption, error) {
	opt := &SearchQueryOption{
		ConsistencyLevel: entity.ClBounded, // default
	}
	info, ok := c.cache.getCollectionInfo(collName)
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
		ts, ok := c.cache.getSessionTs(collName)
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

// IsL0 specifies it is to import L0 segment binlog
func IsL0(isL0 bool) BulkInsertOption {
	return func(req *milvuspb.ImportRequest) {
		optionMap := entity.KvPairsMap(req.GetOptions())
		optionMap["l0_import"] = strconv.FormatBool(isL0)
		req.Options = entity.MapKvPairs(optionMap)
	}
}

// SkipDiskQuotaCheck https://github.com/milvus-io/milvus/pull/35274
func SkipDiskQuotaCheck(skipDiskQuotaCheck bool) BulkInsertOption {
	return func(req *milvuspb.ImportRequest) {
		optionMap := entity.KvPairsMap(req.GetOptions())
		optionMap["skip_disk_quota_check"] = strconv.FormatBool(skipDiskQuotaCheck)
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

type listCollectionOpt struct {
	showInMemory bool
}

type ListCollectionOption func(*listCollectionOpt)

func WithShowInMemory(value bool) ListCollectionOption {
	return func(opt *listCollectionOpt) {
		opt.showInMemory = value
	}
}

type DropCollectionOption func(*milvuspb.DropCollectionRequest)

type ReleaseCollectionOption func(*milvuspb.ReleaseCollectionRequest)

type FlushOption func(*milvuspb.FlushRequest)

type createDatabaseOpt struct {
	Base       *commonpb.MsgBase
	Properties map[string]string
}

type CreateDatabaseOption func(*createDatabaseOpt)

func WithDatabaseProperty(key, value string) CreateDatabaseOption {
	return func(opt *createDatabaseOpt) {
		if opt.Properties == nil {
			opt.Properties = make(map[string]string, 0)
		}
		opt.Properties[key] = value
	}
}

type DropDatabaseOption func(*milvuspb.DropDatabaseRequest)

type DescribeDatabaseOption func(*milvuspb.DescribeDatabaseRequest)

type ReplicateMessageOption func(*milvuspb.ReplicateMessageRequest)

type CreatePartitionOption func(*milvuspb.CreatePartitionRequest)

type DropPartitionOption func(*milvuspb.DropPartitionRequest)

type LoadPartitionsOption func(*milvuspb.LoadPartitionsRequest)

type ReleasePartitionsOption func(*milvuspb.ReleasePartitionsRequest)

// CreateResourceGroupOption is an option that is used in CreateResourceGroup API.
type CreateResourceGroupOption func(*milvuspb.CreateResourceGroupRequest)

// WithCreateResourceGroupConfig returns a CreateResourceGroupOption that setup the config.
func WithCreateResourceGroupConfig(config *entity.ResourceGroupConfig) CreateResourceGroupOption {
	return func(req *milvuspb.CreateResourceGroupRequest) {
		req.Config = config
	}
}

// UpdateResourceGroupsOption is an option that is used in UpdateResourceGroups API.
type UpdateResourceGroupsOption func(*milvuspb.UpdateResourceGroupsRequest)

// WithUpdateResourceGroupConfig returns an UpdateResourceGroupsOption that sets the new config to the specified resource group.
func WithUpdateResourceGroupConfig(resourceGroupName string, config *entity.ResourceGroupConfig) UpdateResourceGroupsOption {
	return func(urgr *milvuspb.UpdateResourceGroupsRequest) {
		if urgr.ResourceGroups == nil {
			urgr.ResourceGroups = make(map[string]*entity.ResourceGroupConfig)
		}
		urgr.ResourceGroups[resourceGroupName] = config
	}
}
