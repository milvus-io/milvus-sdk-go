## Milvus Go SDK

Go SDK for [Milvus](https://github.com/milvus-io/milvus). To contribute code to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) first.

|Milvus version| Recommended Go SDK version |
|:-----:|:-----:|
| 1.0.x | 1.0.1 |
| 1.1.x | 1.1.0 |

### Getting started

#### Prerequisites

Go 1.12 or higher

#### Install Milvus Go SDK

1. Use `go get` to install the latest version of the Milvus Go SDK and dependencies:

   ```shell
   go get -u github.com/milvus-io/milvus-sdk-go/milvus
   ```

2. Include the Milvus Go SDK in your application:

   ```shell
   import "github.com/milvus-io/milvus-sdk-go/milvus"
   ```

#### Try an example

```shell
cd [milvus-sdk-go root path]/examples
go run MilvusClientExample.go
```

### API Documentation

Refer to [https://godoc.org/github.com/milvus-io/milvus-sdk-go/milvus](https://godoc.org/github.com/milvus-io/milvus-sdk-go/milvus) for the GO SDK API documentation.

### Code format

The Go source code is formatted using gofmt and golint.
