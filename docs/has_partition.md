# Has Partition

API to check whether partition from collection with specified name exists

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, collection name of the partition;

- `partitionName` string, partition name to check;

## Response

- `result` bool, stands for whether the partition exists

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
has, err := cli.HasPartition(ctx, "TestCollection", "Partition1")
// handles the error not nil
```
