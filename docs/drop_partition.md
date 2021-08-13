# Drop Partition

API to drop partition from collection by name

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, collection name of the partition

- `partitionName` string, partition name to drop;

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.DropPartition(ctx, "TestCollection", "Partition1")
// handles the error not nil
```
