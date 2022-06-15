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
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
)

// CreateCollectionOption is an option that is used to modify CreateCollectionRequest
type CreateCollectionOption func(*server.CreateCollectionRequest)

// WithConsistencyLevel specifies a specific ConsistencyLevel, rather than using the default ReaderProperties.
func WithConsistencyLevel(cl entity.ConsistencyLevel) CreateCollectionOption {
	return func(req *server.CreateCollectionRequest) {
		req.ConsistencyLevel = cl.CommonConsisencyLevel()
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

// SearchOption is an option that is used to modify SearchRequest
type SearchOption func(*server.SearchRequest)

// WithConsistencyLevel specifies a specific ConsistencyLevel, rather than using the default ReaderProperties.
func WithSearchConsistencyLevel(cl entity.ConsistencyLevel) SearchOption {
	return func(req *server.SearchRequest) {
		req.GuaranteeTimestamp = uint64(cl.CommonConsisencyLevel())
	}
}
