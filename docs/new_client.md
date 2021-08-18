# New Client

This is the "Constructor" method set (only one for current release) to obtain `Client` instance.

## Common Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `addr`       | Address of the Milvus instance                               | String                   |


## Response

`err`: connection error (if any).

## Example

```go
ctx := context.Background()
cli, err := client.NewGrpcClient(ctx, "localhost:19530")
```
