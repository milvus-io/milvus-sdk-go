# Describe Index 

API to describe index on vector field of a collection

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to describe index on

- `fieldName` string, the field name to describe index on 

## Response

- `indexes` slice of entity.Index, the indexes on the vector field specified

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error for field specified is not valid 
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
indexes, err := cli.DescribeIndex(ctx, "TestCollection", "Vector")
// handles the error not nil
```
