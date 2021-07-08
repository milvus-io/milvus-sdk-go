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

// Collection represent collection meta in Milvus
type Collection struct {
	ID               int64   // collection id
	Name             string  // collection name
	Schema           *Schema // collection schema, with fields schema and primary key definition
	PhysicalChannels []string
	VirtualChannels  []string
}

// Partition represent partition meta in Milvus
type Partition struct {
	ID   int64  // paritition id
	Name string // parition name
}
