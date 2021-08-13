# New Client

This is the "Constructor" method set (there is only one for now) to obtain `Client` instance.

## Common Params

- ctx context.Context is the context to control connection process. 

- addr string is the addr of the Milvus instance

## Response

- err contains the connection error (if any) 

```go
ctx := context.Background()
cli, err := client.NewGrpcClient(ctx, "localhost:19530")
```
