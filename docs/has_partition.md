# Has Partition

API to check if a partition from a collection with specified name exists.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection with the partition to check.           | String                   |
| `partitionName` | Name of the partition to check.                            | String                   |


## Response

- `result`: boolean value that stands for whether the partition exists or not.

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
has, err := cli.HasPartition(ctx, "TestCollection", "Partition1")
// handles the error not nil
```
