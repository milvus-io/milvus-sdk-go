# Describe Collection

API to describe collection information 

## Params

- `ctx` context.Context, context to control API invocation process;

- `name` string, collection name to describe

## Response

- `collection` pointer of entity.Collection, represents the collection to describe

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
coll, err := cli.DescribeCollection(ctx, "TestCollection")
// handles the error not nil
```
