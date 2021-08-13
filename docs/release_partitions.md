# Release Partitions

API to release partitions by name

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, collection name to release;

- `paritionNames` slice of string, partition names to release;

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.ReleasePartitions(ctx, "TestCollection",[]string{"Partitions"})
// handles the error not nil
```
