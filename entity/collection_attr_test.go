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

package entity

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/suite"
)

type CollectionTTLSuite struct {
	suite.Suite
}

func (s *CollectionTTLSuite) TestValid() {
	type testCase struct {
		input     string
		expectErr bool
	}

	cases := []testCase{
		{input: "a", expectErr: true},
		{input: "1000", expectErr: false},
		{input: "0", expectErr: false},
		{input: "-10", expectErr: true},
	}

	for _, tc := range cases {
		s.Run(tc.input, func() {
			ca := ttlCollAttr{}
			ca.value = tc.input
			err := ca.Valid()
			if tc.expectErr {
				s.Error(err)
			} else {
				s.NoError(err)
			}
		})
	}
}

func (s *CollectionTTLSuite) TestCollectionTTL() {
	type testCase struct {
		input     int64
		expectErr bool
	}

	cases := []testCase{
		{input: 1000, expectErr: false},
		{input: 0, expectErr: false},
		{input: -10, expectErr: true},
	}

	for _, tc := range cases {
		s.Run(fmt.Sprintf("%d", tc.input), func() {
			ca := CollectionTTL(tc.input)
			key, value := ca.KeyValue()
			s.Equal(cakTTL, key)
			s.Equal(strconv.FormatInt(tc.input, 10), value)
			err := ca.Valid()
			if tc.expectErr {
				s.Error(err)
			} else {
				s.NoError(err)
			}
		})
	}
}

func TestCollectionTTL(t *testing.T) {
	suite.Run(t, new(CollectionTTLSuite))
}
