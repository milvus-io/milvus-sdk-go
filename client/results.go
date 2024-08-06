package client

import (
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// SearchResult contains the result from Search api of client
// IDs is the auto generated id values for the entities
// Fields contains the data of `outputFieleds` specified or all columns if non
// Scores is actually the distance between the vector current record contains and the search target vector
type SearchResult struct {
	ResultCount  int // the returning entry count
	GroupByValue entity.Column
	IDs          entity.Column // auto generated id, can be mapped to the columns from `Insert` API
	Fields       ResultSet     //[]entity.Column // output field data
	Scores       []float32     // distance to the target vector
	Err          error         // search error if any
}

func (sr *SearchResult) Slice(start, end int) *SearchResult {
	id := end

	result := &SearchResult{
		IDs:    sr.IDs.Slice(start, end),
		Fields: sr.Fields.Slice(start, end),

		Err: sr.Err,
	}
	if sr.GroupByValue != nil {
		result.GroupByValue = sr.GroupByValue.Slice(start, end)
	}

	result.ResultCount = result.IDs.Len()

	l := len(sr.Scores)
	if start > l {
		start = l
	}
	if id > l || id < 0 {
		id = l
	}
	result.Scores = sr.Scores[start:id]

	return result
}

// ResultSet is an alias type for column slice.
type ResultSet []entity.Column

func (rs ResultSet) Len() int {
	if len(rs) == 0 {
		return 0
	}
	return rs[0].Len()
}

func (rs ResultSet) Slice(start, end int) ResultSet {
	result := make([]entity.Column, 0, len(rs))
	for _, col := range rs {
		result = append(result, col.Slice(start, end))
	}
	return result
}

// GetColumn returns column with provided field name.
func (rs ResultSet) GetColumn(fieldName string) entity.Column {
	for _, column := range rs {
		if column.Name() == fieldName {
			return column
		}
	}
	return nil
}
