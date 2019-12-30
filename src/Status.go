package main

type ErrorCode int32

const (
	OK			 ErrorCode = 0

	//system error section
	UnKnownError ErrorCode = 1
	NotSupported ErrorCode = 2
	NotConnected ErrorCode = 3

	//function error section
	RPCFailed	 ErrorCode = 4
	ServerFailed ErrorCode = 5
)


type Status interface {
	ok() bool
	getStatus() status
	getMessage() string
}

type status struct {
	ErrorCode	 int32
	state		 string
}

func NewStatus(_status status) Status {
	return &status{_status.ErrorCode, _status.state,}
}

func NewStatus1(error_code ErrorCode, state string) Status {
	return &status{int32(error_code), state,}
}

func (_status status)ok() bool {
	return _status.ErrorCode == int32(OK)
}

func (_status status)getStatus() status {
	return _status
}

func (_status status)getMessage() string {
	return _status.state
}
