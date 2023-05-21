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
	"github.com/cockroachdb/errors"
	"github.com/tidwall/gjson"
)

// ColumnDynamic is a logically wrapper for dynamic json field with provided output field.
type ColumnDynamic struct {
	*ColumnJSONBytes
	outputField string
}

func NewColumnDynamic(column *ColumnJSONBytes, outputField string) *ColumnDynamic {
	return &ColumnDynamic{
		ColumnJSONBytes: column,
		outputField:     outputField,
	}
}

func (c *ColumnDynamic) Name() string {
	return c.outputField
}

func (c *ColumnDynamic) GetInt64(idx int) (int64, error) {
	bs, err := c.ColumnJSONBytes.ValueByIdx(idx)
	if err != nil {
		return 0, err
	}
	r := gjson.GetBytes(bs, c.outputField)
	if !r.Exists() {
		return 0, errors.New("column not has value")
	}
	if r.Type != gjson.Number {
		return 0, errors.New("column not int")
	}
	return r.Int(), nil
}

func (c *ColumnDynamic) GetString(idx int) (string, error) {
	bs, err := c.ColumnJSONBytes.ValueByIdx(idx)
	if err != nil {
		return "", err
	}
	r := gjson.GetBytes(bs, c.outputField)
	if !r.Exists() {
		return "", errors.New("column not has value")
	}
	if r.Type != gjson.String {
		return "", errors.New("column not string")
	}
	return r.String(), nil
}

func (c *ColumnDynamic) GetBool(idx int) (bool, error) {
	bs, err := c.ColumnJSONBytes.ValueByIdx(idx)
	if err != nil {
		return false, err
	}
	r := gjson.GetBytes(bs, c.outputField)
	if !r.Exists() {
		return false, errors.New("column not has value")
	}
	if !r.IsBool() {
		return false, errors.New("column not string")
	}
	return r.Bool(), nil
}

func (c *ColumnDynamic) GetDouble(idx int) (float64, error) {
	bs, err := c.ColumnJSONBytes.ValueByIdx(idx)
	if err != nil {
		return 0, err
	}
	r := gjson.GetBytes(bs, c.outputField)
	if !r.Exists() {
		return 0, errors.New("column not has value")
	}
	if r.Type != gjson.Number {
		return 0, errors.New("column not string")
	}
	return r.Float(), nil
}
