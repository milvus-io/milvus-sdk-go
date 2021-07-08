// Copyright (C) 2019-2021 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// Package client provides milvus client functions
package client

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/entity"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/internal/proto/server"
)

// Index insert into collection with column-based format
// collName is the collection name
// partitionName is the partition to insert, if not specified(empty), default partition will be used
// columns are slice of the column-based data
func (c *grpcClient) Insert(ctx context.Context, collName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
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
	if len(columns) == 0 {
		return nil, errors.New("no column provided")
	}
	req.NumRows = uint32(columns[0].Len())
	for _, column := range columns {
		if uint32(column.Len()) != req.NumRows {
			return nil, errors.New("column number not matched")
		}
		req.FieldsData = append(req.FieldsData, column.FieldData())
	}
	resp, err := c.service.Insert(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	return entity.IDColumns(resp.GetIDs(), 0, -1)
}

// Flush force collection to flush memory records into storage
// in sync mode, flush will wait all segments to be flushed
func (c *grpcClient) Flush(ctx context.Context, collName string, async bool) error {
	if c.service == nil {
		return ErrClientNotReady
	}
	req := &server.FlushRequest{
		DbName:          "", //reserved,
		CollectionNames: []string{collName},
	}
	resp, err := c.service.Flush(ctx, req)
	if err != nil {
		return err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return err
	}
	//TODO silverxia, add context done respect logic while waiting
	if !async {
		segmentIDs, has := resp.GetCollSegIDs()[collName]
		if has {
			waitingSet := make(map[int64]struct{})
			for _, segmentID := range segmentIDs.GetData() {
				fmt.Println("wait", segmentID)
				waitingSet[segmentID] = struct{}{}
			}
			flushed := func() bool {
				segments, err := c.GetPersistentSegmentInfo(context.Background(), collName)
				if err != nil {
					//TODO handles grpc failure, maybe need reconnect?
				}
				flushed := 0
				for _, segment := range segments {
					if _, has := waitingSet[segment.ID]; !has {
						continue
					}
					if !segment.Flushed() {
						return false
					}
					flushed++
				}
				return len(waitingSet) == flushed
			}
			for !flushed() {
				time.Sleep(500 * time.Millisecond)
			}
		}
	}
	return nil
}

//BoolExprSearch search with bool expression
func (c *grpcClient) Search(ctx context.Context, collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, params map[string]string) ([]SearchResult, error) {
	if c.service == nil {
		return []SearchResult{}, ErrClientNotReady
	}
	bs, _ := json.Marshal(params)
	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":  vectorField,
		"topk":        fmt.Sprintf("%d", topK),
		"params":      string(bs),
		"metric_type": string(metricType),
	})

	req := &server.SearchRequest{
		DbName:           "",
		CollectionName:   collName,
		PartitionNames:   []string{},
		SearchParams:     searchParams,
		Dsl:              expr,
		DslType:          common.DslType_BoolExprV1,
		OutputFields:     outputFields,
		PlaceholderGroup: vector2PlaceholderGroupBytes(vectors),
	}
	resp, err := c.service.Search(ctx, req)
	if err != nil {
		return []SearchResult{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []SearchResult{}, err
	}
	results := resp.GetResults()
	offset := 0
	sr := make([]SearchResult, 0, results.GetNumQueries())
	fieldDataList := results.GetFieldsData()
	for i := 0; i < int(results.GetNumQueries()); i++ {

		rc := int(results.GetTopks()[i]) // result entry count for current query
		entry := SearchResult{
			ResultCount: rc,
			Scores:      results.GetScores()[offset:rc],
		}
		entry.IDs, entry.Err = entity.IDColumns(results.GetIds(), offset, offset+rc)
		if entry.Err != nil {
			offset += rc
			continue
		}
		entry.Fields = make([]entity.Column, 0, len(fieldDataList))
		for _, fieldData := range fieldDataList {
			column, err := entity.FieldDataColumn(fieldData, offset, offset+rc)
			if err != nil {
				entry.Err = err
				continue
			}
			entry.Fields = append(entry.Fields, column)
		}
		sr = append(sr, entry)
		offset += rc
	}
	return sr, nil
}

// GetPersistentSegmentInfo get persistent segment info
func (c *grpcClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetPersistentSegmentInfoRequest{
		DbName:         "", //reserved
		CollectionName: collName,
	}
	resp, err := c.service.GetPersistentSegmentInfo(ctx, req)
	if err != nil {
		return []*entity.Segment{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []*entity.Segment{}, err
	}
	segments := make([]*entity.Segment, 0, len(resp.GetInfos()))
	for _, info := range resp.GetInfos() {
		segments = append(segments, &entity.Segment{
			ID:           info.GetSegmentID(),
			CollectionID: info.GetCollectionID(),
			ParititionID: info.GetPartitionID(),
			NumRows:      info.GetNumRows(),
			State:        info.GetState(),
		})
	}

	return segments, nil
}

// GetQuerySegmentInfo get query query cluster segment loaded info
func (c *grpcClient) GetQuerySegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &server.GetQuerySegmentInfoRequest{
		DbName:         "", //reserved
		CollectionName: collName,
	}
	resp, err := c.service.GetQuerySegmentInfo(ctx, req)
	if err != nil {
		return []*entity.Segment{}, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return []*entity.Segment{}, err
	}

	segments := make([]*entity.Segment, 0, len(resp.GetInfos()))
	for _, info := range resp.GetInfos() {
		segments = append(segments, &entity.Segment{
			ID:           info.GetSegmentID(),
			CollectionID: info.GetCollectionID(),
			ParititionID: info.GetPartitionID(),
			IndexID:      info.GetIndexID(),
			NumRows:      info.GetNumRows(),
		})
	}

	return segments, nil
}

func vector2PlaceholderGroupBytes(vectors []entity.Vector) []byte {
	phg := &server.PlaceholderGroup{
		Placeholders: []*server.PlaceholderValue{
			vector2Placeholder(vectors),
		},
	}

	bs, _ := proto.Marshal(phg)
	return bs
}

func vector2Placeholder(vectors []entity.Vector) *server.PlaceholderValue {
	ph := &server.PlaceholderValue{
		Tag:    "$0",
		Type:   server.PlaceholderType_FloatVector,
		Values: make([][]byte, 0, len(vectors)),
	}
	for _, vector := range vectors {
		ph.Values = append(ph.Values, vector.Serialize())
	}
	return ph
}
