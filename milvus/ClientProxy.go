package milvus

import (
	pb "github.com/milvus-io/milvus-sdk-go/milvus/grpc/gen"
	"google.golang.org/grpc"
	"log"
)

type Milvusclient struct {
	MClient MilvusGrpcClient
}

// NewMilvusClient is the constructor of MilvusClient
func NewMilvusClient(client MilvusGrpcClient) MilvusClient {
	return &Milvusclient{client}
}

func (client *Milvusclient) GetClientVersion() string {
	return clientVersion
}

func (client *Milvusclient) Connect(connectParam ConnectParam) Status {
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure())
	opts = append(opts, grpc.WithBlock())

	serverAddr := connectParam.IPAddress + ":" + connectParam.Port

	conn, err := grpc.Dial(serverAddr, opts...)
	if err != nil {
		log.Fatalf("fail to dial: %v", err)
	}

	milvusclient := pb.NewMilvusServiceClient(conn)

	milvusGrpcClient := NewMilvusGrpcClient(milvusclient)

	client.MClient = milvusGrpcClient

	return status{0, ""}
}

func (client *Milvusclient) IsConnected() bool {
	return client.MClient != nil
}

func (client *Milvusclient) Disconnect() Status {
	client.MClient = nil
	return status{0, ""}
}

func (client *Milvusclient) CreateTable(tableSchema TableSchema) Status {
	grpcTableSchema := pb.TableSchema{nil, tableSchema.TableName, tableSchema.Dimension,
		tableSchema.IndexFileSize, int32(tableSchema.MetricType), struct{}{}, nil, 0}
	grpcStatus := client.MClient.CreateTable(grpcTableSchema)
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}
}

func (client *Milvusclient) HasTable(tableName string) bool {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	return client.MClient.HasTable(grpcTableName).BoolReply
}

func (client *Milvusclient) DropTable(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus := client.MClient.DropTable(grpcTableName)
	errorCode := int64(grpcStatus.ErrorCode)
	return status{errorCode, grpcStatus.Reason}
}

func (client *Milvusclient) CreateIndex(indexParam *IndexParam) Status {
	index := pb.Index{int32(indexParam.IndexType), int32(indexParam.Nlist),
		struct{}{}, nil, 0}
	grpcIndexParam := pb.IndexParam{nil, indexParam.TableName, &index,
		struct{}{}, nil, 0}
	grpcStatus := client.MClient.CreateIndex(grpcIndexParam)
	return status{int64(grpcStatus.ErrorCode), grpcStatus.Reason}
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Insert(insertParam *InsertParam) Status {
	var i int
	var rowRecordArray = make([]*pb.RowRecord, len(insertParam.RecordArray))
	for i = 0; i < len(insertParam.RecordArray); i++ {
		rowRecord := pb.RowRecord{insertParam.RecordArray[i], struct{}{}, nil, 0}
		rowRecordArray[i] = &rowRecord
	}
	grpcInsertParam := pb.InsertParam{insertParam.TableName, rowRecordArray, insertParam.IDArray,
		insertParam.PartitionTag, struct{}{}, nil, 0}
	vectorIds := client.MClient.Insert(grpcInsertParam)
	insertParam.IDArray = vectorIds.VectorIdArray
	return status{int64(vectorIds.Status.ErrorCode), vectorIds.Status.Reason}
}

////////////////////////////////////////////////////////////////////////////

func (client *Milvusclient) Search(searchParam SearchParam) (Status, TopkQueryResult) {
	var queryRecordArray = make([]*pb.RowRecord, len(searchParam.QueryVectors))
	var i, j int64
	for i = 0; i < int64(len(searchParam.QueryVectors)); i++ {
		rowRecord := pb.RowRecord{searchParam.QueryVectors[i], struct{}{}, nil, 0}
		queryRecordArray[i] = &rowRecord
	}
	grpcSearchParam := pb.SearchParam{searchParam.TableName, queryRecordArray, nil,
		searchParam.Topk, searchParam.Nprobe, searchParam.PartitionTag, struct{}{}, nil, 0}
	topkQueryResult := client.MClient.Search(grpcSearchParam)
	nq := topkQueryResult.GetRowNum()
	var result = make([]QueryResult, nq)
	for i = 0; i < nq; i++ {
		topk := int64(len(topkQueryResult.GetIds())) / nq
		result[i].Ids = make([]int64, topk)
		result[i].Distances = make([]float32, topk)
		for j = 0; j < topk; j++ {
			result[i].Ids[j] = topkQueryResult.GetIds()[i*nq+j]
			result[i].Distances[j] = topkQueryResult.GetDistances()[i*nq+j]
		}
	}
	return status{int64(topkQueryResult.Status.ErrorCode), topkQueryResult.Status.Reason},
		TopkQueryResult{result}
}

func (client *Milvusclient) DescribeTable(tableName string) (Status, TableSchema) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	tableSchema := client.MClient.DescribeTable(grpcTableName)
	return status{int64(tableSchema.GetStatus().GetErrorCode()), tableSchema.Status.Reason},
		TableSchema{tableSchema.GetTableName(), tableSchema.GetDimension(), tableSchema.GetIndexFileSize(), int64(tableSchema.GetMetricType())}
}

func (client *Milvusclient) CountTable(tableName string) (Status, int64) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	rowCount := client.MClient.CountTable(grpcTableName)
	return status{int64(rowCount.GetStatus().GetErrorCode()), rowCount.GetStatus().GetReason()}, rowCount.GetTableRowCount()

}

func (client *Milvusclient) ShowTables() (Status, []string) {
	tableNameList := client.MClient.ShowTable()
	return status{int64(tableNameList.GetStatus().GetErrorCode()), tableNameList.GetStatus().GetReason()}, tableNameList.GetTableNames()
}

func (client *Milvusclient) ServerVersion() (Status, string) {
	command := pb.Command{"version", struct{}{}, nil, 0}
	serverVersion := client.MClient.Cmd(command)
	return status{int64(serverVersion.GetStatus().GetErrorCode()), serverVersion.GetStatus().GetReason()}, serverVersion.GetStringReply()
}

func (client *Milvusclient) ServerStatus() (Status, string) {
	if client.MClient == nil {
		return status{int64(0), ""}, "not connect to server"
	}
	command := pb.Command{"", struct{}{}, nil, 0}
	serverStatus := client.MClient.Cmd(command)
	return status{int64(serverStatus.GetStatus().GetErrorCode()), serverStatus.GetStatus().GetReason()}, serverStatus.GetStringReply()
}

func (client *Milvusclient) PreloadTable(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus := client.MClient.PreloadTable(grpcTableName)
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) DescribeIndex(tableName string) (Status, IndexParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	indexParam := client.MClient.DescribeIndex(grpcTableName)
	return status{int64(indexParam.GetStatus().GetErrorCode()), indexParam.GetStatus().GetReason()},
		IndexParam{indexParam.GetTableName(), IndexType(indexParam.GetIndex().GetIndexType()), int64(indexParam.GetIndex().GetNlist())}
}

func (client *Milvusclient) DropIndex(tableName string) Status {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcStatus := client.MClient.DropIndex(grpcTableName)
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) CreatePartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus := client.MClient.CreatePartition(grpcPartitionParam)
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.GetReason()}
}

func (client *Milvusclient) ShowPartitions(tableName string) (Status, []PartitionParam) {
	grpcTableName := pb.TableName{tableName, struct{}{}, nil, 0}
	grpcPartitionList := client.MClient.ShowPartitions(grpcTableName)
	var partitionList = make([]PartitionParam, len(grpcPartitionList.GetPartitionArray()))
	var i int
	for i = 0; i < len(grpcPartitionList.GetPartitionArray()); i++ {
		partitionList[i].TableName = grpcPartitionList.GetPartitionArray()[i].TableName
		partitionList[i].PartitionTag = grpcPartitionList.GetPartitionArray()[i].Tag
		partitionList[i].PartitionName = grpcPartitionList.GetPartitionArray()[i].PartitionName
	}
	return status{int64(grpcPartitionList.GetStatus().GetErrorCode()), grpcPartitionList.GetStatus().GetReason()}, partitionList
}

func (client *Milvusclient) DropPartition(partitionParam PartitionParam) Status {
	grpcPartitionParam := pb.PartitionParam{partitionParam.TableName, partitionParam.PartitionName,
		partitionParam.PartitionTag, struct{}{}, nil, 0}
	grpcStatus := client.MClient.DropPartition(grpcPartitionParam)
	return status{int64(grpcStatus.GetErrorCode()), grpcStatus.Reason}
}

func (client *Milvusclient) GetConfig(nodeName string) (Status, string) {
	command := pb.Command{"get_config " + nodeName, struct{}{}, nil, 0}
	configInfo := client.MClient.Cmd(command)
	return status{int64(configInfo.GetStatus().GetErrorCode()), configInfo.GetStatus().GetReason()}, configInfo.GetStringReply()
}

func (client *Milvusclient) SetConfig(nodeName string, value string) Status {
	command := pb.Command{"set_config " + nodeName + " " + value, struct{}{}, nil, 0}
	reply := client.MClient.Cmd(command)
	return status{int64(reply.GetStatus().GetErrorCode()), reply.GetStatus().GetReason()}
}
