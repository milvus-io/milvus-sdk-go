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

package client

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
)

func TestRetryOnMilvusErrors(t *testing.T) {
	ctx := context.Background()

	n := 0
	testFn := func() (interface{}, error) {
		if n < 3 {
			n++
			return &commonpb.Status{
				Code: 0,
			}, nil
		}
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100))
	assert.Equal(t, int32(0), resp.(interface{ GetCode() int32 }).GetCode())
	assert.NoError(t, err)
	t.Log(resp)
}

func TestOnNoCode(t *testing.T) {
	ctx := context.Background()

	n := 0
	testFn := func() (interface{}, error) {
		if n < 3 {
			n++
			return &commonpb.Status{
				Code: 0,
			}, nil
		}
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn)
	assert.Equal(t, int32(0), resp.(interface{ GetCode() int32 }).GetCode())
	assert.NoError(t, err)
	t.Log(resp)
}

func TestReturnErr(t *testing.T) {
	ctx := context.Background()

	testFn := func() (interface{}, error) {
		return nil, errors.New("mock err")
	}

	_, err := RetryOnMilvusErrors(ctx, testFn)
	assert.Error(t, err)
	t.Log(err)
}

func TestAttempts(t *testing.T) {
	ctx := context.Background()

	testFn := func() (interface{}, error) {
		t.Log("executed")
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100), Attempts(1))
	assert.Equal(t, int32(100), resp.(interface{ GetCode() int32 }).GetCode())
	assert.NoError(t, err)
	t.Log(resp)
}

func TestMaxSleepTime(t *testing.T) {
	ctx := context.Background()

	testFn := func() (interface{}, error) {
		t.Log("executed")
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100), Attempts(3), MaxSleepTime(200*time.Millisecond))
	assert.Equal(t, int32(100), resp.(interface{ GetCode() int32 }).GetCode())
	assert.NoError(t, err)
	t.Log(resp)
}

func TestSleep(t *testing.T) {
	ctx := context.Background()

	testFn := func() (interface{}, error) {
		t.Log("executed")
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100), Attempts(3), Sleep(500*time.Millisecond))
	assert.Equal(t, int32(100), resp.(interface{ GetCode() int32 }).GetCode())
	assert.NoError(t, err)
	t.Log(resp)
}

func TestContextDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	testFn := func() (interface{}, error) {
		t.Log("executed")
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100))
	assert.Equal(t, int32(100), resp.(interface{ GetCode() int32 }).GetCode())
	assert.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
	t.Log(resp)
}

func TestContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	testFn := func() (interface{}, error) {
		t.Log("executed")
		return &commonpb.Status{
			Reason: "mock err",
			Code:   100,
		}, nil
	}

	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	resp, err := RetryOnMilvusErrors(ctx, testFn, OnMerrCodes(100))
	assert.Equal(t, int32(100), resp.(interface{ GetCode() int32 }).GetCode())
	assert.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	t.Log(resp)
	t.Log(err)
}
