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
	"errors"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	server "github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
)

// CreateCollectionOption is an option that is used to modify CreateCollectionRequest
type CreateCollectionOption func(*server.CreateCollectionRequest)

// WithConsistencyLevel specifies a specific ConsistencyLevel, rather than using the default ReaderProperties.
func WithConsistencyLevel(cl entity.ConsistencyLevel) CreateCollectionOption {
	return func(req *server.CreateCollectionRequest) {
		req.ConsistencyLevel = cl.CommonConsistencyLevel()
	}
}

// LoadCollectionOption is an option that is used to modify LoadCollectionRequest
type LoadCollectionOption func(*server.LoadCollectionRequest)

// WithReplicaNumber specifies a specific ReplicaNumber, rather than using the default ReplicaNumber.
func WithReplicaNumber(rn int32) LoadCollectionOption {
	return func(req *server.LoadCollectionRequest) {
		req.ReplicaNumber = rn
	}
}

// SearchQueryOption is an option of search/query request
type SearchQueryOption struct {
	ConsistencyLevel   entity.ConsistencyLevel
	GuaranteeTimestamp uint64
	TravelTimestamp    uint64
}

// SearchQueryOptionFunc is a function which modifies SearchOption
type SearchQueryOptionFunc func(option *SearchQueryOption)

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

// WithTravelTimestamp specifies time travel timestamp
func WithTravelTimestamp(tt uint64) SearchQueryOptionFunc {
	return func(option *SearchQueryOption) {
		option.TravelTimestamp = tt
	}
}

func MakeSearchQueryOption(collName string, opts ...SearchQueryOptionFunc) (*SearchQueryOption, error) {
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
