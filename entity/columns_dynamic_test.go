package entity

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type ColumnDynamicSuite struct {
	suite.Suite
}

func (s *ColumnDynamicSuite) TestGetInt() {
	cases := []struct {
		input       string
		expectErr   bool
		expectValue int64
	}{
		{`{"field": 1000000000000000001}`, false, 1000000000000000001},
		{`{"field": 4418489049307132905}`, false, 4418489049307132905},
		{`{"other_field": 4418489049307132905}`, true, 0},
		{`{"field": "string"}`, true, 0},
	}

	for _, c := range cases {
		s.Run(c.input, func() {
			column := NewColumnDynamic(&ColumnJSONBytes{
				values: [][]byte{[]byte(c.input)},
			}, "field")
			v, err := column.GetInt64(0)
			if c.expectErr {
				s.Error(err)
				return
			}
			s.NoError(err)
			s.Equal(c.expectValue, v)
		})
	}
}

func (s *ColumnDynamicSuite) TestGetString() {
	cases := []struct {
		input       string
		expectErr   bool
		expectValue string
	}{
		{`{"field": "abc"}`, false, "abc"},
		{`{"field": "test"}`, false, "test"},
		{`{"other_field": "string"}`, true, ""},
		{`{"field": 123}`, true, ""},
	}

	for _, c := range cases {
		s.Run(c.input, func() {
			column := NewColumnDynamic(&ColumnJSONBytes{
				values: [][]byte{[]byte(c.input)},
			}, "field")
			v, err := column.GetString(0)
			if c.expectErr {
				s.Error(err)
				return
			}
			s.NoError(err)
			s.Equal(c.expectValue, v)
		})
	}
}

func (s *ColumnDynamicSuite) TestGetBool() {
	cases := []struct {
		input       string
		expectErr   bool
		expectValue bool
	}{
		{`{"field": true}`, false, true},
		{`{"field": false}`, false, false},
		{`{"other_field": true}`, true, false},
		{`{"field": "test"}`, true, false},
	}

	for _, c := range cases {
		s.Run(c.input, func() {
			column := NewColumnDynamic(&ColumnJSONBytes{
				values: [][]byte{[]byte(c.input)},
			}, "field")
			v, err := column.GetBool(0)
			if c.expectErr {
				s.Error(err)
				return
			}
			s.NoError(err)
			s.Equal(c.expectValue, v)
		})
	}
}

func (s *ColumnDynamicSuite) TestGetDouble() {
	cases := []struct {
		input       string
		expectErr   bool
		expectValue float64
	}{
		{`{"field": 1}`, false, 1.0},
		{`{"field": 6231.123}`, false, 6231.123},
		{`{"other_field": 1.0}`, true, 0},
		{`{"field": "string"}`, true, 0},
	}

	for _, c := range cases {
		s.Run(c.input, func() {
			column := NewColumnDynamic(&ColumnJSONBytes{
				values: [][]byte{[]byte(c.input)},
			}, "field")
			v, err := column.GetDouble(0)
			if c.expectErr {
				s.Error(err)
				return
			}
			s.NoError(err)
			s.Less(v-c.expectValue, 1e-10)
		})
	}

}

func (s *ColumnDynamicSuite) TestIndexOutOfRange() {
	var err error
	column := NewColumnDynamic(&ColumnJSONBytes{}, "field")

	s.Equal("field", column.Name())

	_, err = column.GetInt64(0)
	s.Error(err)

	_, err = column.GetString(0)
	s.Error(err)

	_, err = column.GetBool(0)
	s.Error(err)

	_, err = column.GetDouble(0)
	s.Error(err)
}

func TestColumnDynamic(t *testing.T) {
	suite.Run(t, new(ColumnDynamicSuite))
}
