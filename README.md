## Milvus Go SDK

### Getting started

#### Prerequisites

Go 1.12 or higher

#### Install Go SDK

1. Use `go get` to install the latest version of the Milvus Go SDK and dependencies:

   ```shell
   go get -u github.com/milvus-io/milvus-sdk-go/milvus
   ```

2. Include the Milvus Go SDK in your application:

   ```shell
   import "github.com/milvus-io/milvus-sdk-go/milvus"
   ```

### Try an example

```shell
cd [milvus-sdk-go root path]/examples
go run MilvusClientExample.go
```

### Code format

The Go source code is formatted using gofmt and golint.
