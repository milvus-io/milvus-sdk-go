package client

import (
	"context"
	"errors"

	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/schema"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
)

// Index insert into collection with column-based format
// collName is the collection name
// partitionName is the partition to insert, if not specified(empty), default partition will be used
// columns are slice of the column-based data
func (c *grpcClient) Insert(ctx context.Context, collName string, partitionName string, columns []entity.Column) (entity.Column, error) {
	if c.service == nil {
		return nil, ErrClientNotReady
	}
	req := &server.InsertRequest{
		DbName:         "", // reserved
		CollectionName: collName,
		PartitionName:  partitionName,
	}
	if req.PartitionName == "" {
		req.PartitionName = "_default" // use default partition
	}
	for _, column := range columns {
		req.FieldsData = append(req.FieldsData, column.FieldData())
	}
	resp, err := c.service.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	var idColumn entity.Column
	switch field := resp.GetIDs().GetIdField().(type) {
	case *schema.IDs_IntId:
		idColumn = entity.NewColumnInt64("", field.IntId.GetData())
	case *schema.IDs_StrId:
		idColumn = entity.NewColumnString("", field.StrId.GetData())
	default:
		return nil, errors.New("unsupported id type")
	}
	return idColumn, nil
}
