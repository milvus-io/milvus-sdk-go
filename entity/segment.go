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

import (
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
)

// Segment represent segment in milvus
type Segment struct {
	ID           int64
	CollectionID int64
	ParititionID int64
	IndexID      int64

	NumRows int64
	State   common.SegmentState
}

// Flushed indicates segment is flushed
func (s Segment) Flushed() bool {
	return s.State == common.SegmentState_Flushed
}
