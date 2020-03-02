/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// package milvus
package milvus

import ()

var clientVersion string = "0.1.0"

// MetricType metric type
type MetricType int64

// IndexType index type
type IndexType int64

const (
	// L2 euclidean distance
	L2 MetricType = 1
	// IP inner product
	IP MetricType = 2
	// HAMMING hamming distance
	HAMMING MetricType = 3
	// JACCARD jaccard distance
	JACCARD MetricType = 4
	// TANIMOTO tanimoto distance
	TANIMOTO MetricType = 5
)

const (
	// INVALID invald index type
	INVALID IndexType = 0
	// FLAT flat
	FLAT IndexType = 1
	// IVFFLAT ivfflat
	IVFFLAT IndexType = 2
	// IVFSQ8 ivfsq8
	IVFSQ8 IndexType = 3
	//RNSG rnsg
	RNSG IndexType = 4
	// IVFSQ8H ivfsq8h
	IVFSQ8H IndexType = 5
	// IVFPQ ivfpq
	IVFPQ IndexType = 6
	// SPTAGKDT sptagkdt
	SPTAGKDT IndexType = 7
	// SPTAGBKT sptagbkt
	SPTAGBKT IndexType = 8
	// HNSW hnsw
	HNSW IndexType = 11
)

// ConnectParam Parameters for connect
type ConnectParam struct {
	// IPAddress Server IP address
	IPAddress string
	// Port Server port
	Port string
}

//TableSchema informations of a table
type TableSchema struct {
	// TableName table name
	TableName string
	// Dimension Vector dimension, must be a positive value
	Dimension int64
	// IndexFileSize Index file size, must be a positive value
	IndexFileSize int64
	// MetricType Index metric type
	MetricType int64
}

// IndexParam index parameters
type IndexParam struct {
	// TableName table name for create index
	TableName string
	// IndexType create index type
	IndexType IndexType
	// Nlist index nlist
	Nlist int64
}

// RowRecord record typy
type RowRecord struct {
	FloatData  []float32
	BinaryData []byte
}

// InsertParam insert parameters
type InsertParam struct {
	// TableName table name
	TableName string
	// PartitionTag partition tag
	PartitionTag string
	// RecordArray raw vectors array
	RecordArray []RowRecord
	// IDArray id array
	IDArray []int64
}

// Range range information, for DATE range, the format is like: 'year-month-day'
type Range struct {
	// StartValue Range start
	StartValue string
	// EndValue Range stop
	EndValue string
}

// SearchParam search parameters
type SearchParam struct {
	// TableName table name for search
	TableName string
	// QueryVectors query vectors raw array
	QueryVectors []RowRecord
	// Topk topk
	Topk int64
	// Nprobe nprobe
	Nprobe int64
	// PartitionTag partition tag array
	PartitionTag []string
}

// SearchByIDParamParam search parameters
type SearchByIDParam struct {
	// TableName table name for search
	TableName string
	// Id vector id
	Id int64
	// Topk topk
	Topk int64
	// Nprobe nprobe
	Nprobe int64
	// PartitionTag partition tag array
	PartitionTag []string
}

//QueryResult Query result
type QueryResult struct {
	// Ids id array
	Ids []int64
	// Distances distance array
	Distances []float32
}

// TopkQueryResult Topk query result
type TopkQueryResult struct {
	// QueryResultList query result list
	QueryResultList []QueryResult
}

// PartitionParam partition parameters
type PartitionParam struct {
	// TableName partition table name
	TableName string
	// PartitionTag partition tag
	PartitionTag string
}

type GetVectorIDsParam struct {
	TableName   string
	SegmentName string
}

// MilvusClient SDK main interface
type MilvusClient interface {

	// GetClientVersion method
	// This method is used to give the client version.
	// return Client version.
	GetClientVersion() (string, error)

	// Connect method
	// Create a connection instance and return it's shared pointer
	// return indicate if connect is successful
	Connect(connectParam ConnectParam) (Status, error)

	// IsConnected method
	// This method is used to test whether server is connected
	// return indicate if connection status
	IsConnected() (bool, error)

	// Disconnect method
	// This method is used to disconnect server
	// return indicate if disconnect is successful
	Disconnect() (Status, error)

	// CreateTable method
	// This method is used to create table
	// param tableSchema is used to provide table information to be created.
	// return indicate if table is created successfully
	CreateTable(tableSchema TableSchema) (Status, error)

	// HasTable method
	// This method is used to create table.
	//return indicate if table is exist
	HasTable(tableName string) (Status, bool, error)

	// DropTable method
	// This method is used to drop table(and its partitions).
	// return indicate if table is drop successfully.
	DropTable(tableName string) (Status, error)

	// CreateIndex method
	// This method is used to create index for whole table(and its partitions).
	// return indicate if build index successfully.
	CreateIndex(indexParam *IndexParam) (Status, error)

	// Insert method
	// This method is used to query vector in table.
	// return indicate if insert is successful.
	Insert(insertParam *InsertParam) (Status, error)

	// GetVectorByID method
	// This method is used to get vector by vector id
	// return vector data
	GetVectorByID(tableName string, vector_id int64) (RowRecord, Status, error)

	// GetVectorIDs method
	// This method is used to get vector ids
	// return vector ids
	GetVectorIDs(getVectorIDsParam GetVectorIDsParam) ([]int64, Status, error)

	// Search method
	// This method is used to query vector in table.
	// return indicate if query is successful.
	Search(searchParam SearchParam) (TopkQueryResult, Status, error)

	// SearchByID method
	// This method is used to search by id
	// return vector data
	SearchByID(searchByIDParam SearchByIDParam) (TopkQueryResult, Status, error)

	// DeleteByID method
	// This method is used to delete vectors by ids
	// return indicate if delete is successful
	DeleteByID(tableName string, id_array []int64) (Status, error)

	// DescribeTable method
	// This method is used to show table information.
	//return indicate if this operation is successful.
	DescribeTable(tableName string) (TableSchema, Status, error)

	// CountTable method
	// This method is used to get table row count.
	// return indicate if this operation is successful.
	CountTable(tableName string) (int64, Status, error)

	// ShowTables method
	// This method is used to list all tables.
	// return indicate if this operation is successful.
	ShowTables() ([]string, Status, error)

	// ServerVersion method
	// This method is used to give the server version.
	// return server version.
	ServerVersion() (string, Status, error)

	// ServerStatus method
	// This method is used to give the server status.
	// return server status.
	ServerStatus() (string, Status, error)

	// PreloadTable method
	// This method is used to preload table
	// return indicate if this operation is successful.
	PreloadTable(tableName string) (Status, error)

	// DescribeIndex method
	// This method is used to describe index
	// return indicate if this operation is successful.
	DescribeIndex(tableName string) (IndexParam, Status, error)

	// DropIndex method
	// This method is used to drop index of table(and its partitions)
	// return indicate if this operation is successful.
	DropIndex(tableName string) (Status, error)

	// CreatePartition method
	// This method is used to create table partition
	// return indicate if partition is created successfully
	CreatePartition(partitionParam PartitionParam) error

	// ShowPartition method
	// This method is used to create table
	// return indicate if this operation is successful
	ShowPartitions(tableName string) ([]PartitionParam, error)

	// DropPartition method
	// This method is used to delete table partition.
	// return indicate if partition is delete successfully.
	DropPartition(partitionParam PartitionParam) error

	// GetConfig
	// This method is used to get config
	// return indicate if this operation is successful.
	GetConfig(nodeName string) (string, error)

	// SetConfig
	// This method is used to set config
	// return indicate if this operation is successful.
	SetConfig(nodeName string, value string) error

	// Flush method
	// This method is used to flush tables
	// return indicate if flush is successful
	Flush(tableNaeArray []string) error

	// Compact method
	// This method is used to compact table
	// return indicate if compact is successful
	Compact(tableName string) error
}
