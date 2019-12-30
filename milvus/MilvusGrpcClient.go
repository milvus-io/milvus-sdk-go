package milvus

import (
	"context"
	"time"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
)

// MilvusGrpcClient call grpc generated code interface
type MilvusGrpcClient interface {
	CreateTable(tableSchema pb.TableSchema) pb.Status

	HasTable(tableName pb.TableName) pb.BoolReply

	DescribeTable(tableName pb.TableName) pb.TableSchema

	CountTable(tableName pb.TableName) pb.TableRowCount

	ShowTable() pb.TableNameList

	DropTable(tableName pb.TableName) pb.Status

	CreateIndex(indexParam pb.IndexParam) pb.Status

	DescribeIndex(tableName pb.TableName) pb.IndexParam

	DropIndex(tableName pb.TableName) pb.Status

	CreatePartition(partitionParam pb.PartitionParam) pb.Status

	ShowPartitions(tableName pb.TableName) pb.PartitionList

	DropPartition(partitionParam pb.PartitionParam) pb.Status

	Insert(insertParam pb.InsertParam) pb.VectorIds

	Search(searchParam pb.SearchParam) *pb.TopKQueryResult

	SearchInFiles(searchInFilesParam pb.SearchInFilesParam) *pb.TopKQueryResult

	Cmd(command pb.Command) pb.StringReply

	DeleteByDate(deleteByDateParam pb.DeleteByDateParam) pb.Status

	PreloadTable(tableName pb.TableName) pb.Status
}

type milvusGrpcClient struct {
	client pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateTable(tableSchema pb.TableSchema) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	reply, err := grpcClient.client.CreateTable(ctx, &tableSchema)
	if err != nil {
		println("CreateTable rpc failed: " + err.Error())
	}
	return *reply
}

func (grpcClient *milvusGrpcClient) HasTable(tableName pb.TableName) pb.BoolReply {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	boolReply, err := grpcClient.client.HasTable(ctx, &tableName)
	if err != nil {

	}

	return *boolReply
}

func (grpcClient *milvusGrpcClient) DescribeTable(tableName pb.TableName) pb.TableSchema {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tableSchema, err := grpcClient.client.DescribeTable(ctx, &tableName)
	if err != nil {
		println("DescribeTable rpc failed: " + err.Error())
	}
	return *tableSchema
}

func (grpcClient *milvusGrpcClient) CountTable(tableName pb.TableName) pb.TableRowCount {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	count, err := grpcClient.client.CountTable(ctx, &tableName)
	if err != nil {
		println("CountTable rpc failed: " + err.Error())
	}
	return *count
}

func (grpcClient *milvusGrpcClient) ShowTable() pb.TableNameList {
	cmd := pb.Command{"", struct{}{}, nil, 0}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tableNameList, err := grpcClient.client.ShowTables(ctx, &cmd)
	if err != nil {
		println("ShowTable rpc failed: " + err.Error())
	}
	return *tableNameList
}

func (grpcClient *milvusGrpcClient) DropTable(tableName pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.DropTable(ctx, &tableName)
	if err != nil {
		println("DropTable rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) CreateIndex(indexParam pb.IndexParam) pb.Status {
	ctx := context.Background()
	status, err := grpcClient.client.CreateIndex(ctx, &indexParam)
	if err != nil {
		println("CreateIndex rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) DescribeIndex(tableName pb.TableName) pb.IndexParam {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	indexParam, err := grpcClient.client.DescribeIndex(ctx, &tableName)
	if err != nil {
		println("DescribeIndex rpc failed: " + err.Error())
	}
	return *indexParam
}

func (grpcClient *milvusGrpcClient) DropIndex(tableName pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.DropIndex(ctx, &tableName)
	if err != nil {
		println("DropIndex rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) CreatePartition(partitionParam pb.PartitionParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.CreatePartition(ctx, &partitionParam)
	if err != nil {
		println("CreatePartition rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) ShowPartitions(tableName pb.TableName) pb.PartitionList {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.ShowPartitions(ctx, &tableName)
	if err != nil {
		println("ShowPartition rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) DropPartition(partitionParam pb.PartitionParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.DropPartition(ctx, &partitionParam)
	if err != nil {
		println("DropPartition rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) Insert(insertParam pb.InsertParam) pb.VectorIds {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	vectorIds, err := grpcClient.client.Insert(ctx, &insertParam)
	if err != nil {
		println("Insert rpc failed: " + err.Error())
	}
	return *vectorIds
}

func (grpcClient *milvusGrpcClient) Search(searchParam pb.SearchParam) *pb.TopKQueryResult {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.client.Search(ctx, &searchParam)
	if err != nil {
		println("Search rpc failed: " + err.Error())
	}
	return topkQueryResult
}

func (grpcClient *milvusGrpcClient) SearchInFiles(searchInFilesParam pb.SearchInFilesParam) *pb.TopKQueryResult {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.client.SearchInFiles(ctx, &searchInFilesParam)
	if err != nil {
		println("SearchInFiles rpc failed: " + err.Error())
	}
	return topkQueryResult
}

func (grpcClient *milvusGrpcClient) Cmd(command pb.Command) pb.StringReply {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stringReply, err := grpcClient.client.Cmd(ctx, &command)
	if err != nil {
		println("Cmd rpc failed: " + err.Error())
	}
	return *stringReply
}

func (grpcClient *milvusGrpcClient) DeleteByDate(deleteByDateParam pb.DeleteByDateParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.DeleteByDate(ctx, &deleteByDateParam)
	if err != nil {
		println("DeleteByDate rpc failed: " + err.Error())
	}
	return *status
}

func (grpcClient *milvusGrpcClient) PreloadTable(tableName pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpcClient.client.PreloadTable(ctx, &tableName)
	if err != nil {
		println("PreloadTable rpc failed: " + err.Error())
	}
	return *status
}
