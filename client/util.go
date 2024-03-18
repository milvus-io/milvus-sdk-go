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
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
)

type config struct {
	attempts     uint
	sleep        time.Duration
	maxSleepTime time.Duration
	merrCodes    []int32
}

func newDefaultConfig() *config {
	return &config{
		attempts:     uint(10),
		sleep:        200 * time.Millisecond,
		maxSleepTime: 3 * time.Second,
	}
}

// Option is used to config the retry function.
type Option func(*config)

// Attempts is used to config the max retry times.
func Attempts(attempts uint) Option {
	return func(c *config) {
		c.attempts = attempts
	}
}

// Sleep is used to config the initial interval time of each execution.
func Sleep(sleep time.Duration) Option {
	return func(c *config) {
		c.sleep = sleep
		// ensure max retry interval is always larger than retry interval
		if c.sleep*2 > c.maxSleepTime {
			c.maxSleepTime = 2 * c.sleep
		}
	}
}

// MaxSleepTime is used to config the max interval time of each execution.
func MaxSleepTime(maxSleepTime time.Duration) Option {
	return func(c *config) {
		// ensure max retry interval is always larger than retry interval
		if c.sleep*2 > maxSleepTime {
			c.maxSleepTime = 2 * c.sleep
		} else {
			c.maxSleepTime = maxSleepTime
		}
	}
}

func OnMerrCodes(codes ...int32) Option {
	return func(c *config) {
		c.merrCodes = append(c.merrCodes, codes...)
	}
}

func contains(codes []int32, target int32) bool {
	for _, c := range codes {
		if c == target {
			return true
		}
	}
	return false
}

func RetryOnMilvusErrors(ctx context.Context, fn func() (interface{}, error), opts ...Option) (interface{}, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	c := newDefaultConfig()
	for _, opt := range opts {
		opt(c)
	}

	if len(c.merrCodes) == 0 {
		return fn()
	}

	var lastResp interface{}
	for i := uint(0); i < c.attempts; i++ {
		resp, err := fn()
		if err != nil {
			return resp, err
		}
		var code int32
		switch r := resp.(type) {
		case interface{ GetStatus() *commonpb.Status }:
			code = r.GetStatus().GetCode()
		case interface{ GetCode() int32 }:
			code = r.GetCode()
		default:
			return resp, nil
		}
		if code == 0 || !contains(c.merrCodes, code) {
			return resp, nil
		}

		deadline, ok := ctx.Deadline()
		if ok && time.Until(deadline) < c.sleep {
			return resp, context.DeadlineExceeded
		}

		lastResp = resp

		select {
		case <-time.After(c.sleep):
		case <-ctx.Done():
			return lastResp, ctx.Err()
		}

		c.sleep *= 2
		if c.sleep > c.maxSleepTime {
			c.sleep = c.maxSleepTime
		}
	}
	return lastResp, nil
}
