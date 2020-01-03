/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package milvus

// ErrorCode error code
type ErrorCode int64

const (
	// OK status
	OK ErrorCode = 0

	// UnKnownError unknow error
	UnKnownError ErrorCode = 1
	// NotSupported not supported operation
	NotSupported ErrorCode = 2
	// NotConnected not connected
	NotConnected ErrorCode = 3

	// RPCFailed rpc failed
	RPCFailed ErrorCode = 4
	// ServerFailed server failed
	ServerFailed ErrorCode = 5
)

// Status for SDK interface return
type Status interface {
	Ok() bool
	GetStatus() status
	GetMessage() string
}

type status struct {
	ErrorCode int64
	state     string
}

// NewStatus constructor of Status
func NewStatus(_status status) Status {
	return &status{_status.ErrorCode, _status.state}
}

// NewStatus1 constructor of Status
func NewStatus1(errorCode ErrorCode, state string) Status {
	return &status{int64(errorCode), state}
}

func (_status status) Ok() bool {
	return _status.ErrorCode == int64(OK)
}

func (_status status) GetStatus() status {
	return _status
}

func (_status status) GetMessage() string {
	return _status.state
}
