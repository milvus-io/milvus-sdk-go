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

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	common "github.com/milvus-io/milvus-proto/go-api/commonpb"
	server "github.com/milvus-io/milvus-proto/go-api/milvuspb"
)

// ref: https://github.com/grpc-ecosystem/go-grpc-middleware

type ctxKey int

const (
	RetryOnRateLimit ctxKey = iota
)

var MaxBackOff = 60 * time.Second

// RetryOnRateLimitInterceptor returns a new retrying unary client interceptor.
func RetryOnRateLimitInterceptor(maxRetry uint, backoffFunc grpc_retry.BackoffFuncContext) grpc.UnaryClientInterceptor {
	return func(parentCtx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if maxRetry == 0 {
			return invoker(parentCtx, method, req, reply, cc, opts...)
		}
		var lastErr error
		for attempt := uint(0); attempt < maxRetry; attempt++ {
			_, err := waitRetryBackoff(parentCtx, attempt, backoffFunc)
			if err != nil {
				return err
			}
			lastErr = invoker(parentCtx, method, req, reply, cc, opts...)
			rspStatus := getResultStatus(reply)
			if retryOnRateLimit(parentCtx) && rspStatus.GetErrorCode() == common.ErrorCode_RateLimit {
				//log.Printf("rate limit retry attempt: %d, backoff for %v, reson: %v\n", attempt, backoff, rspStatus.GetReason())
				continue
			}
			return lastErr
		}
		return lastErr
	}
}

func retryOnRateLimit(ctx context.Context) bool {
	retry, ok := ctx.Value(RetryOnRateLimit).(bool)
	if !ok {
		return true // default true
	}
	return retry
}

// getResultStatus returns status of response.
func getResultStatus(reply interface{}) *common.Status {
	switch r := reply.(type) {
	case *common.Status:
		return r
	case *server.MutationResult:
		return r.GetStatus()
	case *server.BoolResponse:
		return r.GetStatus()
	case *server.SearchResults:
		return r.GetStatus()
	case *server.QueryResults:
		return r.GetStatus()
	case *server.FlushResponse:
		return r.GetStatus()
	default:
		return nil
	}
}

func contextErrToGrpcErr(err error) error {
	switch err {
	case context.DeadlineExceeded:
		return status.Error(codes.DeadlineExceeded, err.Error())
	case context.Canceled:
		return status.Error(codes.Canceled, err.Error())
	default:
		return status.Error(codes.Unknown, err.Error())
	}
}

func waitRetryBackoff(parentCtx context.Context, attempt uint, backoffFunc grpc_retry.BackoffFuncContext) (time.Duration, error) {
	var waitTime time.Duration
	if attempt > 0 {
		waitTime = backoffFunc(parentCtx, attempt)
	}
	if waitTime > 0 {
		if waitTime > MaxBackOff {
			waitTime = MaxBackOff
		}
		timer := time.NewTimer(waitTime)
		select {
		case <-parentCtx.Done():
			timer.Stop()
			return waitTime, contextErrToGrpcErr(parentCtx.Err())
		case <-timer.C:
		}
	}
	return waitTime, nil
}
