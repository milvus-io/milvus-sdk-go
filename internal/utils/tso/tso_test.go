// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tso

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestParseHybridTs(t *testing.T) {
	var ts uint64 = 426152581543231492
	physical, logical := ParseHybridTs(ts)
	assert.EqualValues(t, 1625643087552, physical)
	assert.EqualValues(t, 4, logical)
}

func Test_Tso(t *testing.T) {
	t.Run("test ComposeTSByTime", func(t *testing.T) {
		physical := time.Now()
		logical := int64(1000)
		timestamp := ComposeTSByTime(physical, logical)
		pRes, lRes := ParseTS(timestamp)
		assert.Equal(t, physical.Unix(), pRes.Unix())
		assert.Equal(t, uint64(logical), lRes)
	})

	t.Run("test GetCurrentTime", func(t *testing.T) {
		curTime := GetCurrentTime()
		p, l := ParseTS(curTime)
		subTime := time.Since(p)
		assert.Less(t, subTime, time.Millisecond)
		assert.Equal(t, uint64(0), l)
	})
}
