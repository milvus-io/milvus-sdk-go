# New Client

This is the "Constructor" method set (only one for current release) to obtain `Client` instance.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `addr`       | Address of the Milvus instance                               | String                   |


## Response
- Milvus client instance.
- `err`: connection error (if any).

## Example

```go
ctx := context.Background()
cli, err := client.NewClient(context.Background(), client.Config{
		Address:  "localhost:19530",
	})
```
