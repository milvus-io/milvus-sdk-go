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

package milvus

import ()

var clientVersion string = "0.5.0"

// DataType
type DataType int

// MetricType metric type
type MetricType string

// IndexType index type
type IndexType string

const (
	BOOL 			DataType = 1
	INT32 			DataType = 4
	INT64 			DataType = 5
	FLOAT 			DataType = 10
	DOUBLE 			DataType = 11
	VECTORBINARY 	DataType = 100
	VECTORFLOAT 	DataType = 101
)

const (
	// L2 euclidean distance
	L2 				MetricType = "L2"
	// IP inner product
	IP 				MetricType = "IP"
	// HAMMING hamming distance
	HAMMING 		MetricType = "HAMMING"
	// JACCARD jaccard distance
	JACCARD 		MetricType = "JACCARD"
	// TANIMOTO tanimoto distance
	TANIMOTO 		MetricType = "TANIMOTO"
	// SUBSTRUCTURE substructure distance
	SUBSTRUCTURE 	MetricType = "SUBSTRUCTURE"
	// SUPERSTRUCTURE superstructure
	SUPERSTRUCTURE 	MetricType = "SUPERSTRUCTURE"
)

const (
	// FLAT flat
	FLAT 			IndexType = "FLAT"
	// BINFLAT bin_flat
	BINFLAT 		IndexType = "BIN_FLAT"
	// IVFFLAT ivfflat
	IVFFLAT 		IndexType = "IVF_FLAT"
	// BINIVFFLAT bin_ivf_flat
	BINIVFFLAT 		IndexType = "BIN_IVF_FLAT"

	// IVFSQ8 ivfsq8
	IVFSQ8 			IndexType = "IVF_SQ8"
	//RNSG rnsg
	RNSG 			IndexType = "NSG"
	// IVFSQ8H ivfsq8h
	IVFSQ8H 		IndexType = "IVF_SQ8_HYBRID"
	// IVFPQ ivfpq
	IVFPQ 			IndexType = "IVF_PQ"
	// SPTAGKDT sptagkdt
	SPTAGKDT 		IndexType = "SPTAG_KDT_RNT"
	// SPTAGBKT sptagbkt
	SPTAGBKT 		IndexType = "SPTAG_BKT_RNT"
	// HNSW
	HNSW 			IndexType = "HNSW"
	// ANNOY annoy
	ANNOY 			IndexType = "ANNOY"
)

// ConnectParam Parameters for connect
type ConnectParam struct {
	// IPAddress Server IP address
	IPAddress string
	// Port Server port
	Port string
}

// SegmentStat segment statistics
type SegmentStat struct {
	// SegmentName segment name
	SegmentName string
	// RowCount segment row count
	RowCount int64
	// IndexName index name
	IndexName string
	//DataSize data size
	DataSize int64
}

// PartitionStat
type PartitionStat struct {
	// Tag partition tag
	Tag string
	// RowCount row count of partition
	RowCount int64
	// SegmentsStat array of partition's SegmentStat
	SegmentsStat []SegmentStat
}

// Field
type Field struct {
	FieldName string
	DataType DataType
	IndexParams string
	ExtraParams string
}

// CollectionParam informations of a collection
type Mapping struct {
	// CollectionName collection name
	CollectionName string
	// Fields fields
	Fields []Field
	// ExtraParams extra params
	ExtraParams string
}

// IndexParam index parameters
type IndexParam struct {
	// CollectionName collection name for create index
	CollectionName string
	// FieldName field name for create index
	FieldName string
	// IndexParams string
	// ExtraParams extra parameters
	// 	Note: extra_params is extra parameters list, it must be json format
	//        For different index type, parameter list is different accordingly, for example:
	//        FLAT/IVFLAT/SQ8:  "{nlist: '16384'}"
	//            ///< nlist range:[1, 999999]
	//        IVFPQ:  "{nlist: '16384', m: "12"}"
	//            ///< nlist range:[1, 999999]
	//            ///< m is decided by dim and have a couple of results.
	//        NSG:  "{search_length: '45', out_degree:'50', candidate_pool_size:'300', "knng":'100'}"
	//            ///< search_length range:[10, 300]
	//            ///< out_degree range:[5, 300]
	//            ///< candidate_pool_size range:[50, 1000]
	//            ///< knng range:[5, 300]
	//        HNSW  "{M: '16', efConstruction:'500'}"
	//            ///< M range:[5, 48]
	//            ///< efConstruction range:[topk, 4096]
	IndexParams map[string]interface{}
}

// VectorRowRecord
type VectorRowRecord struct {
	FloatData  []float32
	BinaryData []byte
}

// VectorRecord
type VectorRecord struct {
	VectorRecord []VectorRowRecord
}

// FieldValue
type FieldValue struct {
	FieldName string
	RawData interface{}
}

// Entity
type Entity struct {
	EntityId int64
	Entity map[string]interface{}
}

// InsertParam insert parameters
type InsertParam struct {
	// CollectionName collection name
	CollectionName string
	// RecordArray raw entities array
	Fields []FieldValue
	// IDArray id array
	IDArray []int64
	// PartitionTag partition tag
	PartitionTag string
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
	// CollectionName collection name for search
	CollectionName string
	// QueryEntities query entities raw array
	Dsl map[string]interface{}
	// PartitionTag partition tag array
	PartitionTag []string
}



//QueryResult Query result
type QueryResult struct {
	// Ids id array
	Ids []int64
	// Distances distance array
	Distances []float32
	// Entities
	Entities []Entity
}

// TopkQueryResult Topk query result
type TopkQueryResult struct {
	// QueryResultList query result list
	QueryResultList []QueryResult
}

// PartitionParam partition parameters
type PartitionParam struct {
	// CollectionName partition collection name
	CollectionName string
	// PartitionTag partition tag
	PartitionTag string
}

type ListIDInSegmentParam struct {
	CollectionName string
	SegmentId      int64
}

type CompactParam struct {
	CollectionName string
	threshold float64
}

// MilvusClient SDK main interface
type MilvusClient interface {

	// GetClientVersion method
	// This method is used to give the client version.
	// return Client version.
	GetClientVersion() string

	// Connect method
	// Create a connection instance and return it's shared pointer
	// return indicate if connect is successful
	Connect(connectParam ConnectParam) error

	// IsConnected method
	// This method is used to test whether server is connected
	// return indicate if connection status
	IsConnected() bool

	// Disconnect method
	// This method is used to disconnect server
	// return indicate if disconnect is successful
	Disconnect() error

	// CreateCollection method
	// This method is used to create collection
	// param collectionParam is used to provide collection information to be created.
	// return indicate if collection is created successfully
	CreateCollection(mapping Mapping) (Status, error)

	// HasCollection method
	// This method is used to create collection.
	//return indicate if collection is exist
	HasCollection(collectionName string) (bool, Status, error)

	// DropCollection method
	// This method is used to drop collection(and its partitions).
	// return indicate if collection is drop successfully.
	DropCollection(collectionName string) (Status, error)

	// CreateIndex method
	// This method is used to create index for whole collection(and its partitions).
	// return indicate if build index successfully.
	CreateIndex(indexParam *IndexParam) (Status, error)

	// Insert method
	// This method is used to query entity in collection.
	// return indicate if insert is successful.
	Insert(insertParam InsertParam) ([]int64, Status, error)

	// GetEntityByID method
	// This method is used to get entity by entity id
	// return entity data
	GetEntityByID(collectionName string, fieldName []string, entityId []int64) ([]Entity, Status, error)

	// ListIDInSegment method
	// This method is used to get entity ids
	// return entity ids
	ListIDInSegment(listIDInSegmentParam ListIDInSegmentParam) ([]int64, Status, error)

	// Search method
	// This method is used to query entity in collection.
	// return indicate if query is successful.
	Search(searchParam SearchParam) (TopkQueryResult, Status, error)

	// DeleteEntityByID method
	// This method is used to delete entities by ids
	// return indicate if delete is successful
	DeleteEntityByID(collectionName string, id_array []int64) (Status, error)

	// GetCollectionInfo method
	// This method is used to show collection information.
	//return indicate if this operation is successful.
	GetCollectionInfo(collectionName string) (Mapping, Status, error)

	// CountEntities method
	// This method is used to get collection row count.
	// return indicate if this operation is successful.
	CountEntities(collectionName string) (int64, Status, error)

	// ListCollections method
	// This method is used to list all collections.
	// return indicate if this operation is successful.
	ListCollections() ([]string, Status, error)

	// GetCollectionStats method
	// This method is used to get collection informations
	// return collection informations
	GetCollectionStats(collectionName string) (string, Status, error)

	// ServerVersion method
	// This method is used to give the server version.
	// return server version.
	ServerVersion() (string, Status, error)

	// ServerStatus method
	// This method is used to give the server status.
	// return server status.
	ServerStatus() (string, Status, error)

	// LoadCollection method
	// This method is used to preload collection
	// return indicate if this operation is successful.
	LoadCollection(collectionName string) (Status, error)

	// DropIndex method
	// This method is used to drop index of collection(and its partitions)
	// return indicate if this operation is successful.
	DropIndex(collectionName string, fieldName string) (Status, error)

	// CreatePartition method
	// This method is used to create collection partition
	// return indicate if partition is created successfully
	CreatePartition(partitionParam PartitionParam) (Status, error)

	// ListPartitions method
	// This method is used to create collection
	// return indicate if this operation is successful
	ListPartitions(collectionName string) ([]PartitionParam, Status, error)

	// DropPartition method
	// This method is used to delete collection partition.
	// return indicate if partition is delete successfully.
	DropPartition(partitionParam PartitionParam) (Status, error)

	// GetConfig
	// This method is used to get config
	// return indicate if this operation is successful.
	GetConfig(nodeName string) (string, Status, error)

	// SetConfig
	// This method is used to set config
	// return indicate if this operation is successful.
	SetConfig(nodeName string, value string) (Status, error)

	// Flush method
	// This method is used to flush collections
	// return indicate if flush is successful
	Flush(collectionNameArray []string) (Status, error)

	// Compact method
	// This method is used to compact collection
	// return indicate if compact is successful
	Compact(compactParam CompactParam) (Status, error)
}
