# Flush 

API to flush data inserted 

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to flush;

- `async` bool, switch value of the sync/async behavior, note that the deadline of context is not applied in sync flush

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
err := c.Flush(ctx, "TestCollection", false)
handles the error not nil
```
