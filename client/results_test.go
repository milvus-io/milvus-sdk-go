package client

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/suite"
)

type ResultSetSuite struct {
	suite.Suite
}

func (s *ResultSetSuite) TestResultsetUnmarshal() {
	type MyData struct {
		A int64     `milvus:"name:id"`
		V []float32 `milvus:"name:vector"`
	}
	type OtherData struct {
		A string    `milvus:"name:id"`
		V []float32 `milvus:"name:vector"`
	}

	var (
		idData     = []int64{1, 2, 3}
		vectorData = [][]float32{
			{0.1, 0.2},
			{0.1, 0.2},
			{0.1, 0.2},
		}
	)

	rs := ResultSet([]entity.Column{
		entity.NewColumnInt64("id", idData),
		entity.NewColumnFloatVector("vector", 2, vectorData),
	})
	err := rs.Unmarshal([]MyData{})
	s.Error(err)

	receiver := []MyData{}
	err = rs.Unmarshal(&receiver)
	s.Error(err)

	var ptrReceiver []*MyData
	err = rs.Unmarshal(&ptrReceiver)
	s.NoError(err)

	for idx, row := range ptrReceiver {
		s.Equal(row.A, idData[idx])
		s.Equal(row.V, vectorData[idx])
	}

	var otherReceiver []*OtherData
	err = rs.Unmarshal(&otherReceiver)
	s.Error(err)
}

func (s *ResultSetSuite) TestSearchResultUnmarshal() {
	type MyData struct {
		A int64     `milvus:"name:id"`
		V []float32 `milvus:"name:vector"`
	}
	type OtherData struct {
		A string    `milvus:"name:id"`
		V []float32 `milvus:"name:vector"`
	}

	var (
		idData     = []int64{1, 2, 3}
		vectorData = [][]float32{
			{0.1, 0.2},
			{0.1, 0.2},
			{0.1, 0.2},
		}
	)

	sr := SearchResult{
		sch: entity.NewSchema().
			WithField(entity.NewField().WithName("id").WithIsPrimaryKey(true).WithDataType(entity.FieldTypeInt64)).
			WithField(entity.NewField().WithName("vector").WithDim(2).WithDataType(entity.FieldTypeFloatVector)),
		IDs: entity.NewColumnInt64("id", idData),
		Fields: ResultSet([]entity.Column{
			entity.NewColumnFloatVector("vector", 2, vectorData),
		}),
	}
	err := sr.Unmarshal([]MyData{})
	s.Error(err)

	receiver := []MyData{}
	err = sr.Unmarshal(&receiver)
	s.Error(err)

	var ptrReceiver []*MyData
	err = sr.Unmarshal(&ptrReceiver)
	s.NoError(err)

	for idx, row := range ptrReceiver {
		s.Equal(row.A, idData[idx])
		s.Equal(row.V, vectorData[idx])
	}

	var otherReceiver []*OtherData
	err = sr.Unmarshal(&otherReceiver)
	s.Error(err)
}

func TestResults(t *testing.T) {
	suite.Run(t, new(ResultSetSuite))
}
