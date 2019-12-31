package milvus

import (
	"context"
	"log"
	"time"

	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
)

var timeout time.Duration = 10 * time.Second

// MilvusGrpcClient call grpc generated code interface
type MilvusGrpcClient interface {
	CreateTable(tableSchema pb.TableSchema) (pb.Status, error)

	HasTable(tableName pb.TableName) (pb.BoolReply, error)

	DescribeTable(tableName pb.TableName) (pb.TableSchema, error)

	CountTable(tableName pb.TableName) (pb.TableRowCount, error)

	ShowTable() (pb.TableNameList, error)

	DropTable(tableName pb.TableName) (pb.Status, error)

	CreateIndex(indexParam pb.IndexParam) (pb.Status, error)

	DescribeIndex(tableName pb.TableName) (pb.IndexParam, error)

	DropIndex(tableName pb.TableName) (pb.Status, error)

	CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error)

	ShowPartitions(tableName pb.TableName) (pb.PartitionList, error)

	DropPartition(partitionParam pb.PartitionParam) (pb.Status, error)

	Insert(insertParam pb.InsertParam) (pb.VectorIds, error)

	Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error)

	SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error)

	Cmd(command pb.Command) (pb.StringReply, error)

	DeleteByDate(deleteByDateParam pb.DeleteByDateParam) (pb.Status, error)

	PreloadTable(tableName pb.TableName) (pb.Status, error)
}

type milvusGrpcClient struct {
	client pb.MilvusServiceClient
}

// NewMilvusGrpcClient is the constructor of MilvusGrpcClient
func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpcClient *milvusGrpcClient) CreateTable(tableSchema pb.TableSchema) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	reply, err := grpcClient.client.CreateTable(ctx, &tableSchema)
	if err != nil {
		log.Println("CreateTable rpc failed: " + err.Error())
	}
	return *reply, err
}

func (grpcClient *milvusGrpcClient) HasTable(tableName pb.TableName) (pb.BoolReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	boolReply, err := grpcClient.client.HasTable(ctx, &tableName)
	if err != nil {
		log.Println("HasTable rpc failed: " + err.Error())
	}

	return *boolReply, err
}

func (grpcClient *milvusGrpcClient) DescribeTable(tableName pb.TableName) (pb.TableSchema, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tableSchema, err := grpcClient.client.DescribeTable(ctx, &tableName)
	if err != nil {
		log.Println("DescribeTable rpc failed: " + err.Error())
	}
	return *tableSchema, err
}

func (grpcClient *milvusGrpcClient) CountTable(tableName pb.TableName) (pb.TableRowCount, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	count, err := grpcClient.client.CountTable(ctx, &tableName)
	if err != nil {
		log.Println("CountTable rpc failed: " + err.Error())
	}
	return *count, err
}

func (grpcClient *milvusGrpcClient) ShowTable() (pb.TableNameList, error) {
	cmd := pb.Command{"", struct{}{}, nil, 0}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	tableNameList, err := grpcClient.client.ShowTables(ctx, &cmd)
	if err != nil {
		log.Println("ShowTable rpc failed: " + err.Error())
	}
	return *tableNameList, err
}

func (grpcClient *milvusGrpcClient) DropTable(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.DropTable(ctx, &tableName)
	if err != nil {
		log.Println("DropTable rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreateIndex(indexParam pb.IndexParam) (pb.Status, error) {
	ctx := context.Background()
	status, err := grpcClient.client.CreateIndex(ctx, &indexParam)
	if err != nil {
		log.Println("CreateIndex rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DescribeIndex(tableName pb.TableName) (pb.IndexParam, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	indexParam, err := grpcClient.client.DescribeIndex(ctx, &tableName)
	if err != nil {
		log.Println("DescribeIndex rpc failed: " + err.Error())
	}
	return *indexParam, err
}

func (grpcClient *milvusGrpcClient) DropIndex(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.DropIndex(ctx, &tableName)
	if err != nil {
		log.Println("DropIndex rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) CreatePartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.CreatePartition(ctx, &partitionParam)
	if err != nil {
		log.Println("CreatePartition rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) ShowPartitions(tableName pb.TableName) (pb.PartitionList, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.ShowPartitions(ctx, &tableName)
	if err != nil {
		log.Println("ShowPartition rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) DropPartition(partitionParam pb.PartitionParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.DropPartition(ctx, &partitionParam)
	if err != nil {
		log.Println("DropPartition rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) Insert(insertParam pb.InsertParam) (pb.VectorIds, error) {
	ctx := context.Background()
	vectorIds, err := grpcClient.client.Insert(ctx, &insertParam)
	if err != nil {
		log.Println("Insert rpc failed: " + err.Error())
	}
	return *vectorIds, err
}

func (grpcClient *milvusGrpcClient) Search(searchParam pb.SearchParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.client.Search(ctx, &searchParam)
	if err != nil {
		log.Println("Search rpc failed: " + err.Error())
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) SearchInFiles(searchInFilesParam pb.SearchInFilesParam) (*pb.TopKQueryResult, error) {
	ctx := context.Background()
	topkQueryResult, err := grpcClient.client.SearchInFiles(ctx, &searchInFilesParam)
	if err != nil {
		log.Println("SearchInFiles rpc failed: " + err.Error())
	}
	return topkQueryResult, err
}

func (grpcClient *milvusGrpcClient) Cmd(command pb.Command) (pb.StringReply, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	stringReply, err := grpcClient.client.Cmd(ctx, &command)
	if err != nil {
		log.Println("Cmd rpc failed: " + err.Error())
	}
	return *stringReply, err
}

func (grpcClient *milvusGrpcClient) DeleteByDate(deleteByDateParam pb.DeleteByDateParam) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.DeleteByDate(ctx, &deleteByDateParam)
	if err != nil {
		log.Println("DeleteByDate rpc failed: " + err.Error())
	}
	return *status, err
}

func (grpcClient *milvusGrpcClient) PreloadTable(tableName pb.TableName) (pb.Status, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	status, err := grpcClient.client.PreloadTable(ctx, &tableName)
	if err != nil {
		log.Println("PreloadTable rpc failed: " + err.Error())
	}
	return *status, err
}
