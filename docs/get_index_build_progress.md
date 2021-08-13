# Get Index Build Progress 

API to describe index building progress 

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to show

- `fieldName` string, the field name to check index building progress

## Response

- `total` int64, total records count to build index on

- `index` int64, indexed records count

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error for field specified is not valid 
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
indexes, err := cli.GetIndexBuildProgress(ctx, "TestCollection", "Vector")
// handles the error not nil
```
