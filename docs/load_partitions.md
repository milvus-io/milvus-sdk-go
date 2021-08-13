# Load Partitions

API to load partitions by name

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, collection name to load;

- `paritionNames` slice of string, partition names to load;

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
err := cli.LoadPartitions(ctx, "TestCollection",[]string{"Partitions"}, false)
// handles the error not nil
```
