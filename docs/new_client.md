# New Client

This is the "Constructor" method set (only one for current release) to obtain `Client` instance.

## Common Params

- `ctx` context.Context is the context to control connection process. 

- `addr` string is the address of the Milvus instance.

## Response

- `err` contains the connection error (if any).

## Example

```go
ctx := context.Background()
cli, err := client.NewGrpcClient(ctx, "localhost:19530")
```
