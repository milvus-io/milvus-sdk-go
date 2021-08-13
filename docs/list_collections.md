# List Collections

API to list all the collections in the connected Milvus instance 

## Params

- `ctx` context.Context, context to control API invocation process;

## Response

- `collections` array of entity.Collection, represents the collections in the Milvus instance

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
colls, err := cli.ListCollection(ctx)
// handles the error not nil
for _, coll := range colls {
// process each collection if needed
}
```
