package milvus

import ()

var clientVersion string = "0.1.0"

// MetricType metric type
type MetricType int32

// IndexType index type
type IndexType int32

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
	MetricType int32
}

// IndexParam index parameters
type IndexParam struct {
	// TableName table name for create index
	TableName string
	// IndexType create index type
	IndexType IndexType
	// Nlist index nlist
	Nlist int32
}

// InsertParam insert parameters
type InsertParam struct {
	// TableName table name
	TableName string
	// PartitionTag partition tag
	PartitionTag string
	// RecordArray raw vectors array
	RecordArray [][]float32
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
	QueryVectors [][]float32
	// DateRange date range array
	DateRanges [][]Range
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
	// PartitionName partition name
	PartitionName string
	// PartitionTag partition tag
	PartitionTag string
}

// MilvusClient SDK main interface
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
