package main

import (
	"flag"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/testdata"
	"log"
	pb "milvus/src/grpc/gen"
)

var (
	tls                = flag.Bool("tls", false, "Connection uses TLS if true, else plain TCP")
	caFile             = flag.String("ca_file", "", "The file containing the CA root cert file")
	serverAddr         = flag.String("server_addr", "localhost:19530", "The server address in the format of host:port")
	serverHostOverride = flag.String("server_host_override", "x.test.youtube.com", "The server name use to verify the hostname returned by TLS handshake")
)

type milvusClient struct {
	mClient MilvusGrpcClient
}

func NewMilvusClient(client MilvusGrpcClient) MilvusClient {
	return &milvusClient{client}
}

func (client *milvusClient) GetClientVersion() string {
	return clientVersion
}

func (client *milvusClient) Connect(connect_param ConnectParam) Status {
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

	milvus_client := pb.NewMilvusServiceClient(conn)

	milvus_grpc_client := NewMilvusGrpcClient(milvus_client)

	client.mClient = milvus_grpc_client

	return status{0, "",}
}

func (client *milvusClient) IsConnected() bool {
	return client.mClient != nil
}

func (client *milvusClient) Disconnect() Status {
	client.mClient = nil
	return status{0, "",}
}

func (client *milvusClient) CreateTable(tableSchema TableSchema) Status {
	grpcTableSchema := pb.TableSchema{nil, tableSchema.TableName, tableSchema.Dimension, tableSchema.IndexFileSize, tableSchema.MetricType, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.CreateTable(grpcTableSchema)
	errorCode := int32(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason,}
}

func (client *milvusClient) HasTable(tableName string) bool {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	return client.mClient.HasTable(grpcTableName).BoolReply
}

func (client *milvusClient) DropTable(table_name string) Status {
	grpcTableName := pb.TableName{table_name, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.DropTable(grpcTableName)
	errorCode := int32(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason,}
}

func (client *milvusClient) CreateIndex(indexParam *IndexParam) Status {
	index := pb.Index{int32(indexParam.IndexType), int32(indexParam.Nlist), struct{}{}, nil, 0,}
	grpcIndexParam := pb.IndexParam{nil, indexParam.TableName, &index, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.CreateIndex(grpcIndexParam)
	return status{int32(grpcStatus.ErrorCode), grpcStatus.Reason,}
}

////////////////////////////////////////////////////////////////////////////

func (client *milvusClient) Insert(insertParam *InsertParam) Status {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i< len(insertParam.RecordArray); i++  {
		rowRecord := pb.RowRecord{insertParam.RecordArray[i], struct{}{}, nil, 0,}
		rowRecordArray[i] = &rowRecord
	}
	grpcInsertParam := pb.InsertParam{insertParam.TableName, rowRecordArray, insertParam.IdArray, insertParam.PartitionTag, struct{}{}, nil, 0,}
	vectorIds := client.mClient.Insert(grpcInsertParam)
	insertParam.IdArray = vectorIds.VectorIdArray
	return status{int32(vectorIds.Status.ErrorCode), vectorIds.Status.Reason,}
}

////////////////////////////////////////////////////////////////////////////

func (client *milvusClient) Search(searchParam SearchParam) (Status, TopkQueryResult) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryVectors))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryVectors)); i++ {
		rowRecord := pb.RowRecord{searchParam.QueryVectors[i], struct{}{}, nil, 0,}
		queryRecordArray[i] = &rowRecord
	}
	grpcSearchParam := pb.SearchParam{searchParam.TableName, queryRecordArray, nil, searchParam.Topk, searchParam.Nprobe, searchParam.PartitionTag, struct{}{}, nil, 0}
	topkQueryResult := client.mClient.Search(grpcSearchParam)
	nq := topkQueryResult.GetRowNum()
	var result = make([]QueryResult, nq)
	for i = 0; i < nq; i++ {
		topk := int64(len(topkQueryResult.GetIds())) / nq
		result[i].Ids = make([]int64, topk)
		result[i].Distances = make([]float32, topk)
		for j = 0; j < topk; j++ {
			result[i].Ids[j] = topkQueryResult.GetIds()[i * nq + j]
			result[i].Distances[j] = topkQueryResult.GetDistances()[i * nq + j]
		}
	}
	return status{int32(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason}, TopkQueryResult{result,}
}

func (client *milvusClient) DescribeTable(tableName string) (Status, TableSchema) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	tableSchema := client.mClient.DescribeTable(grpcTableName)
	return status{int32(tableSchema.GetStatus().GetErrorCode()), tableSchema.Status.Reason,}, TableSchema{tableSchema.GetTableName(), tableSchema.GetDimension(), tableSchema.GetIndexFileSize(), tableSchema.GetMetricType(),}
}

func (client *milvusClient) CountTable(tableName string) (Status, int64) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	rowCount := client.mClient.CountTable(grpcTableName)
	return status{int32(rowCount.GetStatus().GetErrorCode()), rowCount.GetStatus().GetReason(),}, rowCount.GetTableRowCount()

}

func (client *milvusClient) ShowTables() (Status, []string) {
	tableNameList := client.mClient.ShowTable()
	return status{int32(tableNameList.GetStatus().GetErrorCode()), tableNameList.GetStatus().GetReason()}, tableNameList.GetTableNames()
}

func (client *milvusClient) ServerVersion() (Status, string) {
	command := pb.Command{"version", struct{}{}, nil, 0,}
	serverVersion := client.mClient.Cmd(command)
	return status{int32(serverVersion.GetStatus().GetErrorCode()), serverVersion.GetStatus().GetReason()}, serverVersion.GetStringReply()
}

func (client *milvusClient) ServerStatus() (Status, string) {
	command := pb.Command{"", struct{}{}, nil, 0,}
	serverStatus := client.mClient.Cmd(command)
	return status{int32(serverStatus.GetStatus().GetErrorCode()), serverStatus.GetStatus().GetReason()}, serverStatus.GetStringReply()
}

func (client *milvusClient) PreloadTable(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.PreloadTable(grpcTableName)
	return status{int32(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *milvusClient) DescribeIndex(tableName string) (Status, IndexParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	indexParam := client.mClient.DescribeIndex(grpcTableName)
	return status{int32(indexParam.GetStatus().GetErrorCode()), indexParam.GetStatus().GetReason()},
	IndexParam{indexParam.GetTableName(), IndexType(indexParam.GetIndex().GetIndexType()), indexParam.GetIndex().GetNlist()}
}

func (client *milvusClient) DropIndex(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus := client.mClient.DropIndex(grpcTableName)
	return status{int32(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *milvusClient) CreatePartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName, partitionParam.PartitionTag, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.CreatePartition(grpcPartitionParam)
	return status{int32(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *milvusClient) ShowPartitions(tableName string) (Status, []PartitionParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0,}
	grpcPartitionList := client.mClient.ShowPartitions(grpcTableName)
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionArray()); i++ {
		partitionList[i].TableName = grpcPartitionList.GetPartitionArray()[i].TableName
		partitionList[i].PartitionTag = grpcPartitionList.GetPartitionArray()[i].Tag
		partitionList[i].PartitionName = grpcPartitionList.GetPartitionArray()[i].PartitionName
	}
	return status{int32(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, partitionList
}

func (client *milvusClient) DropPartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName, partitionParam.PartitionTag, struct{}{}, nil, 0,}
	grpcStatus := client.mClient.DropPartition(grpcPartitionParam)
	return status{int32(grpcStatus.GetErrorCode()), grpcStatus.Reason,}
}
