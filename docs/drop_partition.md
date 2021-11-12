# Drop Partition

API to drop a partition from a specified collection by name.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to drop partition from.                | String                   |
| `partitionName` | Name of the partition to drop.                             | String                   |

## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error stands for the client is not connected

  - `ErrCollectionNotExists`, error stands for collection of the specified name does not exist

  - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.DropPartition(ctx, "TestCollection", "Partition1")
// handles the error not nil
```
