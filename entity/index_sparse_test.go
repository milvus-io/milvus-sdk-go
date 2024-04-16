package entity

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"
)

type SparseIndexSuite struct {
	suite.Suite
}

func (s *SparseIndexSuite) TestSparseInverted() {
	s.Run("bad_drop_ratio", func() {
		_, err := NewIndexSparseInverted(IP, -1)
		s.Error(err)

		_, err = NewIndexSparseInverted(IP, 1.0)
		s.Error(err)
	})

	s.Run("normal_case", func() {
		idx, err := NewIndexSparseInverted(IP, 0.2)
		s.Require().NoError(err)

		s.Equal("SparseInverted", idx.Name())
		s.Equal(SparseInverted, idx.IndexType())
		params := idx.Params()

		s.Equal("SPARSE_INVERTED_INDEX", params[tIndexType])
		s.Equal("IP", params[tMetricType])
		paramsVal, has := params[tParams]
		s.True(has)
		m := make(map[string]string)
		err = json.Unmarshal([]byte(paramsVal), &m)
		s.Require().NoError(err)
		dropRatio, ok := m["drop_ratio_build"]
		s.True(ok)
		s.Equal("0.2", dropRatio)
	})

	s.Run("search_param", func() {
		_, err := NewIndexSparseInvertedSearchParam(-1)
		s.Error(err)
		_, err = NewIndexSparseInvertedSearchParam(1.0)
		s.Error(err)

		sp, err := NewIndexSparseInvertedSearchParam(0.2)
		s.Require().NoError(err)
		s.EqualValues(0.2, sp.Params()["drop_ratio_search"])
	})
}

func (s *SparseIndexSuite) TestSparseWAND() {
	s.Run("bad_drop_ratio", func() {
		_, err := NewIndexSparseWAND(IP, -1)
		s.Error(err)

		_, err = NewIndexSparseWAND(IP, 1.0)
		s.Error(err)
	})

	s.Run("normal_case", func() {
		idx, err := NewIndexSparseWAND(IP, 0.2)
		s.Require().NoError(err)

		s.Equal("SparseWAND", idx.Name())
		s.Equal(SparseWAND, idx.IndexType())
		params := idx.Params()

		s.Equal("SPARSE_WAND", params[tIndexType])
		s.Equal("IP", params[tMetricType])
		paramsVal, has := params[tParams]
		s.True(has)
		m := make(map[string]string)
		err = json.Unmarshal([]byte(paramsVal), &m)
		s.Require().NoError(err)
		dropRatio, ok := m["drop_ratio_build"]
		s.True(ok)
		s.Equal("0.2", dropRatio)
	})

	s.Run("search_param", func() {
		_, err := NewIndexSparseWANDSearchParam(-1)
		s.Error(err)
		_, err = NewIndexSparseWANDSearchParam(1.0)
		s.Error(err)

		sp, err := NewIndexSparseWANDSearchParam(0.2)
		s.Require().NoError(err)
		s.EqualValues(0.2, sp.Params()["drop_ratio_search"])
	})
}

func TestSparseIndex(t *testing.T) {
	suite.Run(t, new(SparseIndexSuite))
}
