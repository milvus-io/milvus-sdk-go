# Create Index 

API to create index on vector field of a collection

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to build index on 

- `fieldName` string, the field name to build index on 

- `idx` entity.Index, the index definition struct

- `async` bool, switch value of the sync/async behavior, note that the deadline of context is not applied in sync create process

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - error for field specified is not valid 
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
idx, err := entity.NewIndexIvfFlat(entity.L2, 2)
// handles err if the newly created index is not valid
// cli is a valid Client instance
err := cli.CreateIndex(ctx, "TestCollection", "Vector", idx, false)
// handles the error not nil
```
