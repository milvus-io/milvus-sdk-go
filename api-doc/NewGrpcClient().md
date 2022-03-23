# NewGrpcClient()

This is the constructor method set (only one for current release) to obtain a `Client` instance.

## Invocation

```go
client.NewGrpcClient(ctx, addr)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `addr`       | Address of the Milvus instance                               | String                   |


## Return

A Milvus client instance.

## Errors

Connection error (if any).

## Example

```go
milvusClient, err := client.NewGrpcClient(
    context.Background(), 
    "localhost:19530"
)
if err != nil {
  log.Fatal("failed to connect to Milvus:", err.Error())
}
```
