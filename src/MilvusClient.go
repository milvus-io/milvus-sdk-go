package main

import (
)

var clientVersion string = "0.1.0"

type MetricType int32
type IndexType 	int32

const (
	L2 MetricType = 1
	IP MetricType = 2
)

const (
	INVALID  IndexType = 0
	FLAT	 IndexType = 1
	IVFFLAT  IndexType = 2
	IVFSQ8   IndexType = 3
	RNSG	 IndexType = 4
	IVFSQ8H	 IndexType = 5
	IVFPQ	 IndexType = 6
	SPTAGKDT IndexType = 7
	SPTAGBKT IndexType = 8
)

type ConnectParam struct {
	IpAddress string
	Port 	  string
}

type TableSchema struct {
	TableName 	  string
	Dimension 	  int64
	IndexFileSize int64
	MetricType 	  int32
}

type IndexParam struct {
	TableName string
	IndexType IndexType
	Nlist	  int32
}

type InsertParam struct {
	TableName 	 string
	PartitionTag string
	RecordArray  [][]float32
	IdArray		 []int64
}

type Range struct {
	StartValue string
	EndValue   string
}

type SearchParam struct {
	TableName 	 string
	QueryVectors [][]float32
	DateRanges   [][]Range
	Topk		 int64
	Nprobe		 int64
	PartitionTag []string
}

type QueryResult struct {
	Ids 	  []int64
	Distances []float32
}

type TopkQueryResult struct {
	QueryResultList []QueryResult
}

type PartitionParam struct {
	TableName 	  string
	PartitionName string
	PartitionTag  string
}

type MilvusClient interface {

	/** @return the current Milvus client version */
	GetClientVersion() string

	/**
	 *
	 */
	Connect(connectParam ConnectParam) Status

	IsConnected() bool

	Disconnect() Status

	CreateTable(tableSchema TableSchema) Status

	HasTable(tableName string) bool

	DropTable(tableName string) Status

	CreateIndex(indexParam *IndexParam) Status

	Insert(insertParam *InsertParam) Status

	Search(searchParam SearchParam) (Status, TopkQueryResult)

	DescribeTable(tableName string) (Status, TableSchema)

	CountTable(tableName string) (Status, int64)

	ShowTables() (Status, []string)

	ServerVersion() (Status, string)

	ServerStatus() (Status, string)

	PreloadTable(tableName string) Status

	DescribeIndex(tableName string) (Status, IndexParam)

	DropIndex(tableName string) Status

	CreatePartition(partitionParam PartitionParam) Status

	ShowPartitions(tableName string) (Status, []PartitionParam)

	DropPartition(partitionParam PartitionParam) Status
}