package main

import (
	"context"
	"flag"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/testdata"
	"log"
	"time"

	"fmt"
	pb "milvus/src/grpc/gen"
)

type MilvusGrpcClient interface {
	CreateTable(table_schema pb.TableSchema) pb.Status

	HasTable(table_name pb.TableName) pb.BoolReply

	DescribeTable(table_name pb.TableName) pb.TableSchema

	CountTable(table_name pb.TableName) pb.TableRowCount

	ShowTable() pb.TableNameList

	DropTable(table_name pb.TableName) pb.Status

	CreateIndex(index_param pb.IndexParam) pb.Status

	DescribeIndex(table_name pb.TableName) pb.IndexParam

	DropIndex(table_name pb.TableName) pb.Status

	CreatePartition(partition_param pb.PartitionParam) pb.Status

	ShowPartitions(table_name pb.TableName) pb.PartitionList

	DropPartition(partition_param pb.PartitionParam) pb.Status

	Insert(insert_param pb.InsertParam) pb.VectorIds

	Search(search_param pb.SearchParam) *pb.TopKQueryResult

	SearchInFiles(search_in_files_param pb.SearchInFilesParam) *pb.TopKQueryResult

	Cmd(command pb.Command) pb.StringReply

	DeleteByDate(delete_by_date_param pb.DeleteByDateParam) pb.Status

	PreloadTable(table_name pb.TableName) pb.Status
}

type milvusGrpcClient struct {
	client pb.MilvusServiceClient
}


func NewMilvusGrpcClient(client pb.MilvusServiceClient) MilvusGrpcClient {
	return &milvusGrpcClient{client}
}

func (grpc_client *milvusGrpcClient) CreateTable(table_schema pb.TableSchema) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	reply, err := grpc_client.client.CreateTable(ctx, &table_schema)
	if err != nil {

	}
	return *reply
}

func (grpc_client *milvusGrpcClient) HasTable(table_name pb.TableName) pb.BoolReply {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	bool_reply, err := grpc_client.client.HasTable(ctx, &table_name)
	if err != nil {

	}

	return *bool_reply
}

func (grpc_client *milvusGrpcClient) DescribeTable(table_name pb.TableName) pb.TableSchema {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	table_schema, err := grpc_client.client.DescribeTable(ctx, &table_name)
	if err != nil {

	}
	return *table_schema
}

func (grpc_client *milvusGrpcClient) CountTable(table_name pb.TableName) pb.TableRowCount {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	count, err := grpc_client.client.CountTable(ctx, &table_name)
	if err != nil {

	}
	return *count
}

func (grpc_client *milvusGrpcClient) ShowTable() pb.TableNameList {
	cmd := pb.Command{"", struct{}{}, nil, 0,}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	table_name_list, err := grpc_client.client.ShowTables(ctx,&cmd)
	if err != nil {

	}
	return *table_name_list
}

func (grpc_client *milvusGrpcClient) DropTable(table_name pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.DropTable(ctx, &table_name)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) CreateIndex(index_param pb.IndexParam) pb.Status {
	ctx := context.Background()
	status, err := grpc_client.client.CreateIndex(ctx, &index_param)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) DescribeIndex(table_name pb.TableName) pb.IndexParam {
	ctx, cancel := context.WithTimeout(context.Background(),10*time.Second)
	defer cancel()
	index_param, err := grpc_client.client.DescribeIndex(ctx, &table_name)
	if err != nil {

	}
	return *index_param
}

func (grpc_client *milvusGrpcClient) DropIndex(table_name pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.DropIndex(ctx, &table_name)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) CreatePartition(partition_param pb.PartitionParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.CreatePartition(ctx, &partition_param)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) ShowPartitions(table_name pb.TableName) pb.PartitionList {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.ShowPartitions(ctx, &table_name)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) DropPartition(partition_param pb.PartitionParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.DropPartition(ctx, &partition_param)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) Insert(insert_param pb.InsertParam) pb.VectorIds {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	vector_ids, err := grpc_client.client.Insert(ctx, &insert_param)
	if err != nil {

	}
	return *vector_ids
}

func (grpc_client *milvusGrpcClient) Search(search_param pb.SearchParam) *pb.TopKQueryResult {
	ctx := context.Background()
	topk_query_result, err := grpc_client.client.Search(ctx, &search_param)
	if err != nil {

	}
	return topk_query_result
}

func (grpc_client *milvusGrpcClient) SearchInFiles(search_in_files_param pb.SearchInFilesParam) *pb.TopKQueryResult {
	ctx := context.Background()
	topk_query_result, err := grpc_client.client.SearchInFiles(ctx, &search_in_files_param)
	if err != nil {

	}
	return topk_query_result
}

func (grpc_client *milvusGrpcClient) Cmd(command pb.Command) pb.StringReply {
	ctx, cancel:= context.WithTimeout(context.Background(),10*time.Second)
	defer cancel()
	string_reply, err := grpc_client.client.Cmd(ctx, &command)
	if err != nil {

	}
	return *string_reply
}

func (grpc_client *milvusGrpcClient) DeleteByDate(delete_by_date_param pb.DeleteByDateParam) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.DeleteByDate(ctx, &delete_by_date_param)
	if err != nil {

	}
	return *status
}

func (grpc_client *milvusGrpcClient) PreloadTable(table_name pb.TableName) pb.Status {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, err := grpc_client.client.PreloadTable(ctx, &table_name)
	if err != nil {

	}
	return *status
}

func main() {
	a := pb.TableName{"", struct{}{}, nil, 9,}
	fmt.Println(a)

	flag.Parse()
	var opts []grpc.DialOption
	if *tls {
		if *caFile == "" {
			*caFile = testdata.Path("ca.pem")
		}
		creds, err := credentials.NewClientTLSFromFile(*caFile, *serverHostOverride)
		if err != nil {
			log.Fatalf("Failed to create TLS credentials %v", err)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithInsecure())
	}

	opts = append(opts, grpc.WithBlock())
	conn, err := grpc.Dial(*serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()

	client := pb.NewMilvusServiceClient(conn)

	//test
	table_schema := pb.TableSchema{nil, "test_go", 128, 1024, 0, struct{}{}, nil, 0,}
	//CreateTable(client, table_schema)

}
