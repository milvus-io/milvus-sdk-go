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
	"fmt"
)

// ErrServiceFailed indicates error returns from milvus Service
type ErrServiceFailed error

var (
	//ErrClientNotReady error indicates client not ready
	ErrClientNotReady = errors.New("client not ready")
	//ErrStatusNil error indicates response has nil status
	ErrStatusNil = errors.New("response status is nil")
)

// ErrCollectionNotExists indicates the collection with specified collection name does not exist
type ErrCollectionNotExists struct {
	collName string
}

// Error implement error
func (e ErrCollectionNotExists) Error() string {
	return fmt.Sprintf("collection %s does not exist", e.collName)
}

// ErrPartitionNotExists indicates the partition of collection does not exist
type ErrPartitionNotExists struct {
	collName     string
	paritionName string
}

// Error implement error
func (e ErrPartitionNotExists) Error() string {
	return fmt.Sprintf("partition %s of collection %s does not exist", e.paritionName, e.collName)
}

func collNotExistsErr(collName string) ErrCollectionNotExists {
	return ErrCollectionNotExists{collName: collName}
}

func partNotExistsErr(collName, partitionName string) ErrPartitionNotExists {
	return ErrPartitionNotExists{collName: collName, paritionName: partitionName}
}
