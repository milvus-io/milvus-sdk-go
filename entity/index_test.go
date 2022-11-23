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
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGenericIndex(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	name := fmt.Sprintf("generic_index_%d", rand.Int())
	gi := NewGenericIndex(name, IvfFlat, map[string]string{
		tMetricType: string(IP),
	})
	assert.Equal(t, name, gi.Name())
	assert.EqualValues(t, IvfFlat, gi.Params()[tIndexType])
}
