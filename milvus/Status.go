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
