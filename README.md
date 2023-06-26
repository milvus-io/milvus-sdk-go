# Milvus Go SDK

[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/milvus-sdk-go/blob/master/LICENSE)
[![Mergify Status][mergify-status]][mergify]
[![Go Reference](https://pkg.go.dev/badge/github.com/milvus-io/milvus-sdk-go/v2.svg)](https://pkg.go.dev/github.com/milvus-io/milvus-sdk-go/v2)

[mergify]: https://mergify.io
[mergify-status]: https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/milvus-io/milvus-sdk-go&style=plastic


Go SDK for [Milvus](https://github.com/milvus-io/milvus). To contribute code to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) first.


## SDK versions

|Milvus version| Recommended Go SDK version |
|:-----:|:-----:|
| 2.2.x | [2.2.2](https://github.com/milvus-io/milvus-sdk-go/tree/v2.2.2) |
| 2.1.0 | [2.1.0](https://github.com/milvus-io/milvus-sdk-go/tree/v2.1.0) |
| 2.0.0 | [2.0.0](https://github.com/milvus-io/milvus-sdk-go/tree/v2.0.0) |
| 1.1.x | [1.1.0](https://github.com/milvus-io/milvus-sdk-go/tree/v1.1.0) |
| 1.0.x | [1.0.0](https://github.com/milvus-io/milvus-sdk-go/tree/v1.0.0) |

Note: Major versions is NOT compatible between Milvus and SDK

## Getting started

### Prerequisites

Go 1.15 or higher

### Install Milvus Go SDK

1. Use `go get` to install the latest version of the Milvus Go SDK and dependencies:

   ```shell
   go get -u github.com/milvus-io/milvus-sdk-go/v2
   ```

2. Include the Milvus Go SDK in your application:

   ```go
   import "github.com/milvus-io/milvus-sdk-go/v2/client"

   //...other snippet ...
   client, err := client.NewClient(context.Background(), client.Config{
      Address: "localhost:19530",
   })
   if err != nil {
       // handle error
   }
   defer client.Close()

   client.HasCollection(context.Background(), "YOUR_COLLECTION_NAME")
   ```

### API Documentation

Refer to [https://milvus.io/api-reference/go/v2.2.x/About.md](https://milvus.io/api-reference/go/v2.2.x/About.md) for the Go SDK API documentation.

### Examples
   
See [examples](examples/README.md) about how to use this package to communicate with Milvus

## Code format

The Go source code is formatted using gofmt and golint.
