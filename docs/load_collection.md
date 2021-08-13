# Load Collection

API to load collection by name

## Params

- `ctx` context.Context, context to control API invocation process;

- `name` string, collection name to load;

- `async` bool, switch value of the sync/async behavior, note that the deadline of context is not applied in sync load

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.LoadCollection(ctx, "TestCollection", false)
// handles the error not nil
```
