# Drop Index 

API to drop index from vector field of a collection

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to drop index from

- `fieldName` string, the field name to drop index from

## Response

- `err` error of the drop process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error for field specified is not valid 
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
err := cli.DropIndex(ctx, "TestCollection", "Vector")
// handles the error not nil
```
