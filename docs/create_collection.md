# Create Collection

API to create collection according to the schema specified

## Params

- `ctx` context.Context, context to control API invocation process;

- `collSchema` pointer of entity.Schema, the collection schema definition;

- `shardNum` int32, the shard number of the collection to create; If the `shardNum` is set to 0, default shard number will be used, which is 2

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - error for collection with same name already exists
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
schema := &entity.Schema{
// omit for simpliciy
} 

// cli is a valid Client instance
err := cli.CreateCollection(ctx, schema, 1)
// handles the error not nil
```
