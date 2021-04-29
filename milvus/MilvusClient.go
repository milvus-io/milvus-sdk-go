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

import (
	"context"
)

var clientVersion string = "1.1.0"

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
	// SUBSTRUCTURE substructure distance
	SUBSTRUCTURE MetricType = 6
	// SUPERSTRUCTURE superstructure
	SUPERSTRUCTURE MetricType = 7
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
	// ANNOY annoy
	ANNOY IndexType = 12
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

//CollectionParam informations of a collection
type CollectionParam struct {
	// CollectionName collection name
	CollectionName string
	// Dimension Entity dimension, must be a positive value
	Dimension int64
	// IndexFileSize Index file size, must be a positive value
	IndexFileSize int64
	// MetricType Index metric type
	MetricType int32
}

// IndexParam index parameters
type IndexParam struct {
	// CollectionName collection name for create index
	CollectionName string
	// IndexType create index type
	IndexType IndexType
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
	ExtraParams string
}

// Entity record typy
type Entity struct {
	FloatData  []float32
	BinaryData []byte
}

// InsertParam insert parameters
type InsertParam struct {
	// CollectionName collection name
	CollectionName string
	// PartitionTag partition tag
	PartitionTag string
	// RecordArray raw entities array
	RecordArray []Entity
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
	// CollectionName collection name for search
	CollectionName string
	// QueryEntities query entities raw array
	QueryEntities []Entity
	// Topk topk
	Topk int64
	// PartitionTag partition tag array
	PartitionTag []string
	// ExtraParams extra parameters
	//  Note: extra_params is extra parameters list, it must be json format, for example:
	//	 	  For different index type, parameter list is different accordingly
	//		  FLAT/IVFLAT/SQ8/IVFPQ:  "{nprobe: '32'}"
	//			  ///< nprobe range:[1,999999]
	// 		  NSG:  "{search_length:'100'}
	//	 	 	  ///< search_length range:[10, 300]
	//		  HNSW  "{ef: '64'}
	//		 	  ///< ef range:[k, 4096]
	ExtraParams string
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
	// CollectionName partition collection name
	CollectionName string
	// PartitionTag partition tag
	PartitionTag string
}

type ListIDInSegmentParam struct {
	CollectionName string
	SegmentName    string
}

type LoadCollectionParam struct {
	CollectionName   string
	PartitionTagList []string
}

// MilvusClient SDK main interface
type MilvusClient interface {

	// GetClientVersion method
	// This method is used to give the client version.
	// return Client version.
	GetClientVersion(ctx context.Context) string

	// Connect method
	// Create a connection instance and return it's shared pointer
	// return indicate if connect is successful
	Connect(ctx context.Context, connectParam ConnectParam) error

	// IsConnected method
	// This method is used to test whether server is connected
	// return indicate if connection status
	IsConnected(ctx context.Context) bool

	// Disconnect method
	// This method is used to disconnect server
	// return indicate if disconnect is successful
	Disconnect(ctx context.Context) error

	// CreateCollection method
	// This method is used to create collection
	// param collectionParam is used to provide collection information to be created.
	// return indicate if collection is created successfully
	CreateCollection(ctx context.Context, collectionParam CollectionParam) (Status, error)

	// HasCollection method
	// This method is used to create collection.
	//return indicate if collection is exist
	HasCollection(ctx context.Context, collectionName string) (bool, Status, error)

	// DropCollection method
	// This method is used to drop collection(and its partitions).
	// return indicate if collection is drop successfully.
	DropCollection(ctx context.Context, collectionName string) (Status, error)

	// CreateIndex method
	// This method is used to create index for whole collection(and its partitions).
	// return indicate if build index successfully.
	CreateIndex(ctx context.Context, indexParam *IndexParam) (Status, error)

	// Insert method
	// This method is used to query entity in collection.
	// return indicate if insert is successful.
	Insert(ctx context.Context, insertParam *InsertParam) ([]int64, Status, error)

	// GetEntityByID method
	// This method is used to get entity by entity id
	// return entity data
	GetEntityByID(ctx context.Context, collectionName string, partitionTag string, entity_id []int64) ([]Entity, Status, error)

	// ListIDInSegment method
	// This method is used to get entity ids
	// return entity ids
	ListIDInSegment(ctx context.Context, listIDInSegmentParam ListIDInSegmentParam) ([]int64, Status, error)

	// Search method
	// This method is used to query entity in collection.
	// return indicate if query is successful.
	Search(sctx context.Context, earchParam SearchParam) (TopkQueryResult, Status, error)

	// DeleteEntityByID method
	// This method is used to delete entities by ids
	// return indicate if delete is successful
	DeleteEntityByID(ctx context.Context, collectionName string, partitionTag string, id_array []int64) (Status, error)

	// GetCollectionInfo method
	// This method is used to show collection information.
	//return indicate if this operation is successful.
	GetCollectionInfo(ctx context.Context, collectionName string) (CollectionParam, Status, error)

	// CountEntities method
	// This method is used to get collection row count.
	// return indicate if this operation is successful.
	CountEntities(ctx context.Context, collectionName string) (int64, Status, error)

	// ListCollections method
	// This method is used to list all collections.
	// return indicate if this operation is successful.
	ListCollections(ctx context.Context) ([]string, Status, error)

	// GetCollectionStats method
	// This method is used to get collection informations
	// return collection informations
	GetCollectionStats(ctx context.Context, collectionName string) (string, Status, error)

	// ServerVersion method
	// This method is used to give the server version.
	// return server version.
	ServerVersion(ctx context.Context) (string, Status, error)

	// ServerStatus method
	// This method is used to give the server status.
	// return server status.
	ServerStatus(ctx context.Context) (string, Status, error)

	// LoadCollection method
	// This method is used to preload collection
	// return indicate if this operation is successful.
	LoadCollection(ctx context.Context, param LoadCollectionParam) (Status, error)

	// ReleaseCollection method
	// This method is used to release collection
	// return indicate if this operation is successful.
	ReleaseCollection(ctx context.Context, param LoadCollectionParam) (Status, error)

	// GetIndexInfo method
	// This method is used to describe index
	// return indicate if this operation is successful.
	GetIndexInfo(ctx context.Context, collectionName string) (IndexParam, Status, error)

	// DropIndex method
	// This method is used to drop index of collection(and its partitions)
	// return indicate if this operation is successful.
	DropIndex(ctx context.Context, collectionName string) (Status, error)

	// CreatePartition method
	// This method is used to create collection partition
	// return indicate if partition is created successfully
	CreatePartition(ctx context.Context, partitionParam PartitionParam) (Status, error)

	// ListPartitions method
	// This method is used to create collection
	// return indicate if this operation is successful
	ListPartitions(ctx context.Context, collectionName string) ([]PartitionParam, Status, error)

	// DropPartition method
	// This method is used to delete collection partition.
	// return indicate if partition is delete successfully.
	DropPartition(ctx context.Context, partitionParam PartitionParam) (Status, error)

	// GetConfig
	// This method is used to get config
	// return indicate if this operation is successful.
	GetConfig(ctx context.Context, nodeName string) (string, Status, error)

	// SetConfig
	// This method is used to set config
	// return indicate if this operation is successful.
	SetConfig(ctx context.Context, nodeName string, value string) (Status, error)

	// Flush method
	// This method is used to flush collections
	// return indicate if flush is successful
	Flush(ctx context.Context, collectionNameArray []string) (Status, error)

	// Compact method
	// This method is used to compact collection
	// return indicate if compact is successful
	Compact(ctx context.Context, collectionName string) (Status, error)
}
