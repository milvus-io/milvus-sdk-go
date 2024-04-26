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

package client

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/merr"
)

const (
	offsetKey        = `offset`
	limitKey         = `limit`
	ignoreGrowingKey = `ignore_growing`
	forTuningKey     = `for_tuning`
	groupByKey       = `group_by_field`
)

func (c *GrpcClient) HybridSearch(ctx context.Context, collName string, partitions []string, limit int, outputFields []string, reranker Reranker, subRequests []*ANNSearchRequest, opts ...SearchQueryOptionFunc) ([]SearchResult, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	var schema *entity.Schema
	collInfo, ok := MetaCache.getCollectionInfo(collName)
	if !ok {
		coll, err := c.DescribeCollection(ctx, collName)
		if err != nil {
			return nil, err
		}
		schema = coll.Schema
	} else {
		schema = collInfo.Schema
	}

	sReqs := make([]*milvuspb.SearchRequest, 0, len(subRequests))
	nq := 0
	for _, subRequest := range subRequests {
		r, err := subRequest.getMilvusSearchRequest(collInfo, opts...)
		if err != nil {
			return nil, err
		}
		r.CollectionName = collName
		r.PartitionNames = partitions
		r.OutputFields = outputFields
		nq = len(subRequest.vectors)
		sReqs = append(sReqs, r)
	}

	opt := &SearchQueryOption{}
	for _, o := range opts {
		o(opt)
	}
	params := reranker.GetParams()
	params = append(params, &commonpb.KeyValuePair{Key: limitKey, Value: strconv.FormatInt(int64(limit), 10)})
	params = append(params, &commonpb.KeyValuePair{Key: offsetKey, Value: strconv.FormatInt(int64(opt.Offset), 10)})

	req := &milvuspb.HybridSearchRequest{
		CollectionName:   collName,
		PartitionNames:   partitions,
		Requests:         sReqs,
		OutputFields:     outputFields,
		ConsistencyLevel: commonpb.ConsistencyLevel(collInfo.ConsistencyLevel),
		RankParams:       params,
	}

	result, err := c.Service.HybridSearch(ctx, req)

	err = merr.CheckRPCCall(result, err)
	if err != nil {
		return nil, err
	}

	return c.handleSearchResult(schema, outputFields, nq, result)
}

// Search with bool expression
func (c *GrpcClient) Search(ctx context.Context, collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...SearchQueryOptionFunc) ([]SearchResult, error) {
	if c.Service == nil {
		return []SearchResult{}, ErrClientNotReady
	}
	var schema *entity.Schema
	collInfo, ok := MetaCache.getCollectionInfo(collName)
	if !ok {
		coll, err := c.DescribeCollection(ctx, collName)
		if err != nil {
			return nil, err
		}
		schema = coll.Schema
	} else {
		schema = collInfo.Schema
	}

	option, err := makeSearchQueryOption(collName, opts...)
	if err != nil {
		return nil, err
	}
	// 2. Request milvus Service
	req, err := prepareSearchRequest(collName, partitions, expr, outputFields, vectors, vectorField, metricType, topK, sp, option)
	if err != nil {
		return nil, err
	}

	resp, err := c.Service.Search(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}
	// 3. parse result into result
	return c.handleSearchResult(schema, outputFields, len(vectors), resp)
}

func (c *GrpcClient) handleSearchResult(schema *entity.Schema, outputFields []string, nq int, resp *milvuspb.SearchResults) ([]SearchResult, error) {
	sr := make([]SearchResult, 0, nq)
	// parse result into result
	results := resp.GetResults()
	offset := 0
	fieldDataList := results.GetFieldsData()
	gb := results.GetGroupByFieldValue()

	for i := 0; i < int(results.GetNumQueries()); i++ {
		rc := int(results.GetTopks()[i]) // result entry count for current query
		entry := SearchResult{
			ResultCount: rc,
			Scores:      results.GetScores()[offset : offset+rc],
		}

		entry.IDs, entry.Err = entity.IDColumns(schema, results.GetIds(), offset, offset+rc)
		if entry.Err != nil {
			continue
		}
		// parse group-by values
		if gb != nil {
			entry.GroupByValue, entry.Err = entity.FieldDataColumn(gb, offset, offset+rc)
			if entry.Err != nil {
				offset += rc
				continue
			}
		}
		entry.Fields, entry.Err = c.parseSearchResult(schema, outputFields, fieldDataList, i, offset, offset+rc)
		sr = append(sr, entry)

		offset += rc
	}
	return sr, nil
}

func (c *GrpcClient) parseSearchResult(sch *entity.Schema, outputFields []string, fieldDataList []*schemapb.FieldData, _, from, to int) ([]entity.Column, error) {
	var wildcard bool
	outputFields, wildcard = expandWildcard(sch, outputFields)
	// duplicated name will have only one column now
	outputSet := make(map[string]struct{})
	for _, output := range outputFields {
		outputSet[output] = struct{}{}
	}
	// fields := make(map[string]*schemapb.FieldData)
	columns := make([]entity.Column, 0, len(outputFields))
	var dynamicColumn *entity.ColumnJSONBytes
	for _, fieldData := range fieldDataList {
		column, err := entity.FieldDataColumn(fieldData, from, to)
		if err != nil {
			return nil, err
		}
		if fieldData.GetIsDynamic() {
			var ok bool
			dynamicColumn, ok = column.(*entity.ColumnJSONBytes)
			if !ok {
				return nil, errors.New("dynamic field not json")
			}

			// return json column only explicitly specified in output fields and not in wildcard mode
			if _, ok := outputSet[fieldData.GetFieldName()]; !ok && !wildcard {
				continue
			}
		}

		// remove processed field
		delete(outputSet, fieldData.GetFieldName())

		columns = append(columns, column)
	}

	if len(outputSet) > 0 && dynamicColumn == nil {
		var extraFields []string
		for output := range outputSet {
			extraFields = append(extraFields, output)
		}
		return nil, errors.Newf("extra output fields %v found and result does not dynamic field", extraFields)
	}
	// add dynamic column for extra fields
	for outputField := range outputSet {
		column := entity.NewColumnDynamic(dynamicColumn, outputField)
		columns = append(columns, column)
	}

	return columns, nil
}

func expandWildcard(schema *entity.Schema, outputFields []string) ([]string, bool) {
	wildcard := false
	for _, outputField := range outputFields {
		if outputField == "*" {
			wildcard = true
		}
	}
	if !wildcard {
		return outputFields, false
	}

	set := make(map[string]struct{})
	result := make([]string, 0, len(schema.Fields))
	for _, field := range schema.Fields {
		result = append(result, field.Name)
		set[field.Name] = struct{}{}
	}

	// add dynamic fields output
	for _, output := range outputFields {
		if output == "*" {
			continue
		}
		_, ok := set[output]
		if !ok {
			result = append(result, output)
		}
	}
	return result, true
}

func PKs2Expr(backName string, ids entity.Column) string {
	var expr string
	var pkName = ids.Name()
	if ids.Name() == "" {
		pkName = backName
	}
	switch ids.Type() {
	case entity.FieldTypeInt64:
		expr = fmt.Sprintf("%s in %s", pkName, strings.Join(strings.Fields(fmt.Sprint(ids.FieldData().GetScalars().GetLongData().GetData())), ","))
	case entity.FieldTypeVarChar:
		data := ids.FieldData().GetScalars().GetData().(*schemapb.ScalarField_StringData).StringData.GetData()
		for i := range data {
			data[i] = fmt.Sprintf("\"%s\"", data[i])
		}
		expr = fmt.Sprintf("%s in [%s]", pkName, strings.Join(data, ","))
	}
	return expr
}

// Get grabs the inserted entities using the primary key from the Collection.
func (c *GrpcClient) Get(ctx context.Context, collectionName string, ids entity.Column, opts ...GetOption) (ResultSet, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	o := &getOption{}
	for _, opt := range opts {
		opt(o)
	}

	if len(o.outputFields) == 0 {
		coll, err := c.DescribeCollection(ctx, collectionName)
		if err != nil {
			return nil, err
		}
		for _, f := range coll.Schema.Fields {
			o.outputFields = append(o.outputFields, f.Name)
		}
	}

	return c.QueryByPks(ctx, collectionName, o.partitionNames, ids, o.outputFields)
}

// QueryByPks query record by specified primary key(s)
func (c *GrpcClient) QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, outputFields []string, opts ...SearchQueryOptionFunc) (ResultSet, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	// check primary keys
	if ids.Len() == 0 {
		return nil, errors.New("ids len must not be zero")
	}
	if ids.Type() != entity.FieldTypeInt64 && ids.Type() != entity.FieldTypeVarChar { // string key not supported yet
		return nil, errors.New("only int64 and varchar column can be primary key for now")
	}

	expr := PKs2Expr("", ids)

	return c.Query(ctx, collectionName, partitionNames, expr, outputFields, opts...)
}

// Query performs query by expression.
func (c *GrpcClient) Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, opts ...SearchQueryOptionFunc) (ResultSet, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}

	var sch *entity.Schema
	collInfo, ok := MetaCache.getCollectionInfo(collectionName)
	if !ok {
		coll, err := c.DescribeCollection(ctx, collectionName)
		if err != nil {
			return nil, err
		}
		sch = coll.Schema
	} else {
		sch = collInfo.Schema
	}

	option, err := makeSearchQueryOption(collectionName, opts...)
	if err != nil {
		return nil, err
	}

	req := &milvuspb.QueryRequest{
		DbName:             "", // reserved field
		CollectionName:     collectionName,
		Expr:               expr,
		OutputFields:       outputFields,
		PartitionNames:     partitionNames,
		GuaranteeTimestamp: option.GuaranteeTimestamp,
	}
	if option.Offset > 0 {
		req.QueryParams = append(req.QueryParams, &commonpb.KeyValuePair{Key: offsetKey, Value: strconv.FormatInt(option.Offset, 10)})
	}
	if option.Limit > 0 {
		req.QueryParams = append(req.QueryParams, &commonpb.KeyValuePair{Key: limitKey, Value: strconv.FormatInt(option.Limit, 10)})
	}
	if option.IgnoreGrowing {
		req.QueryParams = append(req.QueryParams, &commonpb.KeyValuePair{Key: ignoreGrowingKey, Value: strconv.FormatBool(option.IgnoreGrowing)})
	}

	resp, err := c.Service.Query(ctx, req)
	if err != nil {
		return nil, err
	}
	err = handleRespStatus(resp.GetStatus())
	if err != nil {
		return nil, err
	}

	fieldsData := resp.GetFieldsData()

	columns, err := c.parseSearchResult(sch, outputFields, fieldsData, 0, 0, -1) //entity.FieldDataColumn(fieldData, 0, -1)
	if err != nil {
		return nil, err
	}

	return columns, nil
}

func getPKField(schema *entity.Schema) *entity.Field {
	for _, f := range schema.Fields {
		if f.PrimaryKey {
			return f
		}
	}
	return nil
}

func getVectorField(schema *entity.Schema) *entity.Field {
	for _, f := range schema.Fields {
		if f.DataType == entity.FieldTypeFloatVector || f.DataType == entity.FieldTypeBinaryVector {
			return f
		}
	}
	return nil
}

func prepareSearchRequest(collName string, partitions []string,
	expr string, outputFields []string, vectors []entity.Vector, vectorField string,
	metricType entity.MetricType, topK int, sp entity.SearchParam, opt *SearchQueryOption) (*milvuspb.SearchRequest, error) {
	params := sp.Params()
	params[forTuningKey] = opt.ForTuning
	bs, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}

	searchParams := entity.MapKvPairs(map[string]string{
		"anns_field":     vectorField,
		"topk":           fmt.Sprintf("%d", topK),
		"params":         string(bs),
		"metric_type":    string(metricType),
		"round_decimal":  "-1",
		ignoreGrowingKey: strconv.FormatBool(opt.IgnoreGrowing),
		offsetKey:        fmt.Sprintf("%d", opt.Offset),
		groupByKey:       opt.GroupByField,
	})

	req := &milvuspb.SearchRequest{
		DbName:             "",
		CollectionName:     collName,
		PartitionNames:     partitions,
		Dsl:                expr,
		PlaceholderGroup:   vector2PlaceholderGroupBytes(vectors),
		DslType:            commonpb.DslType_BoolExprV1,
		OutputFields:       outputFields,
		SearchParams:       searchParams,
		GuaranteeTimestamp: opt.GuaranteeTimestamp,
		Nq:                 int64(len(vectors)),
	}
	return req, nil
}

// GetPersistentSegmentInfo get persistent segment info
func (c *GrpcClient) GetPersistentSegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.Service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &milvuspb.GetPersistentSegmentInfoRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.GetPersistentSegmentInfo(ctx, req)
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
func (c *GrpcClient) GetQuerySegmentInfo(ctx context.Context, collName string) ([]*entity.Segment, error) {
	if c.Service == nil {
		return []*entity.Segment{}, ErrClientNotReady
	}
	req := &milvuspb.GetQuerySegmentInfoRequest{
		DbName:         "", // reserved
		CollectionName: collName,
	}
	resp, err := c.Service.GetQuerySegmentInfo(ctx, req)
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

func (c *GrpcClient) CalcDistance(ctx context.Context, collName string, partitions []string,
	metricType entity.MetricType, opLeft, opRight entity.Column) (entity.Column, error) {
	if c.Service == nil {
		return nil, ErrClientNotReady
	}
	if opLeft == nil || opRight == nil {
		return nil, errors.New("operators cannot be nil")
	}

	// check meta
	if err := c.checkCollectionExists(ctx, collName); err != nil {
		return nil, err
	}
	for _, partition := range partitions {
		if err := c.checkPartitionExists(ctx, collName, partition); err != nil {
			return nil, err
		}
	}
	if err := c.checkCollField(ctx, collName, opLeft.Name(), isVectorField); err != nil {
		return nil, err
	}
	if err := c.checkCollField(ctx, collName, opRight.Name(), isVectorField); err != nil {
		return nil, err
	}

	req := &milvuspb.CalcDistanceRequest{
		OpLeft:  columnToVectorsArray(collName, partitions, opLeft),
		OpRight: columnToVectorsArray(collName, partitions, opRight),
		Params: entity.MapKvPairs(map[string]string{
			"metric": string(metricType),
		}),
	}
	if req.OpLeft == nil || req.OpRight == nil {
		return nil, errors.New("invalid operator passed")
	}

	resp, err := c.Service.CalcDistance(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := handleRespStatus(resp.GetStatus()); err != nil {
		return nil, err
	}

	if fd := resp.GetFloatDist(); fd != nil {
		return entity.NewColumnFloat("distance", fd.GetData()), nil
	}
	if id := resp.GetIntDist(); id != nil {
		return entity.NewColumnInt32("distance", id.GetData()), nil
	}

	return nil, errors.New("distance field not supported")
}

func columnToVectorsArray(collName string, partitions []string, column entity.Column) *milvuspb.VectorsArray {
	result := &milvuspb.VectorsArray{}
	switch column.Type() {
	case entity.FieldTypeInt64: // int64 id
		int64Column, ok := column.(*entity.ColumnInt64)
		if !ok {
			return nil // server shall report error
		}
		ids := &milvuspb.VectorIDs{
			CollectionName: collName,
			PartitionNames: partitions,
			FieldName:      column.Name(), // TODO use field name or column name?
			IdArray: &schemapb.IDs{
				IdField: &schemapb.IDs_IntId{
					IntId: &schemapb.LongArray{
						Data: int64Column.Data(),
					},
				},
			},
		}
		result.Array = &milvuspb.VectorsArray_IdArray{IdArray: ids}
	case entity.FieldTypeString: // string id
		stringColumn, ok := column.(*entity.ColumnString)
		if !ok {
			return nil
		}
		ids := &milvuspb.VectorIDs{
			CollectionName: collName,
			PartitionNames: partitions,
			FieldName:      column.Name(),
			IdArray: &schemapb.IDs{
				IdField: &schemapb.IDs_StrId{
					StrId: &schemapb.StringArray{
						Data: stringColumn.Data(),
					},
				},
			},
		}
		result.Array = &milvuspb.VectorsArray_IdArray{IdArray: ids}
	case entity.FieldTypeFloatVector:
		fvColumn, ok := column.(*entity.ColumnFloatVector)
		if !ok {
			return nil
		}
		fvdata := fvColumn.Data()
		data := make([]float32, 0, fvColumn.Len()*fvColumn.Dim())
		for _, row := range fvdata {
			data = append(data, row...)
		}
		result.Array = &milvuspb.VectorsArray_DataArray{DataArray: &schemapb.VectorField{
			Dim: int64(fvColumn.Dim()),
			Data: &schemapb.VectorField_FloatVector{
				FloatVector: &schemapb.FloatArray{
					Data: data,
				},
			},
		}}
	case entity.FieldTypeBinaryVector:
		bvColumn, ok := column.(*entity.ColumnBinaryVector)
		if !ok {
			return nil
		}
		bvdata := bvColumn.Data()
		data := make([]byte, 0, bvColumn.Dim()*bvColumn.Len()/8)
		for _, row := range bvdata {
			data = append(data, row...)
		}
		result.Array = &milvuspb.VectorsArray_DataArray{DataArray: &schemapb.VectorField{
			Dim: int64(bvColumn.Dim()),
			Data: &schemapb.VectorField_BinaryVector{
				BinaryVector: data,
			},
		}}
	default:
		return nil
	}
	return result
}

func isCollectionPrimaryKey(coll *entity.Collection, column entity.Column) bool {
	if coll == nil || coll.Schema == nil || column == nil {
		return false
	}

	// temporary check logic, since only one primary field is supported
	for _, field := range coll.Schema.Fields {
		if field.PrimaryKey {
			if field.Name == column.Name() && field.DataType == column.Type() {
				return true
			}
			return false
		}
	}
	return false
}

// estRowSize estimate size per row for the specified schema
func estRowSize(sch *entity.Schema, selected []string) int64 {
	var total int64
	for _, field := range sch.Fields {
		if len(selected) > 0 {
			found := false
			for _, sel := range selected {
				if field.Name == sel {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		switch field.DataType {
		case entity.FieldTypeBool:
			total++
		case entity.FieldTypeInt8:
			total++
		case entity.FieldTypeInt16:
			total += 2
		case entity.FieldTypeInt32:
			total += 4
		case entity.FieldTypeInt64:
			total += 8
		case entity.FieldTypeFloat:
			total += 4
		case entity.FieldTypeDouble:
			total += 8
		case entity.FieldTypeString:
			// TODO string need varchar[max] syntax like limitation
		case entity.FieldTypeVarChar:
			maxLength, err := strconv.Atoi(field.TypeParams[entity.TypeParamMaxLength])
			if err != nil {
				log.Fatalf("got invalid varchar max length = %s", field.TypeParams[entity.TypeParamMaxLength])
			}
			total += int64(maxLength)
		case entity.FieldTypeFloatVector:
			dimStr := field.TypeParams[entity.TypeParamDim]
			dim, _ := strconv.ParseInt(dimStr, 10, 64)
			total += 4 * dim
		case entity.FieldTypeBinaryVector:
			dimStr := field.TypeParams[entity.TypeParamDim]
			dim, _ := strconv.ParseInt(dimStr, 10, 64)
			total += 4 * dim / 8
		}
	}
	return total
}
