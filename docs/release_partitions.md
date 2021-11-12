# Release Partitions

API to release partitions by names.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to release partitions from.            | String                   |
| `partitionNames` | List of names of the partitions to release.                | Slice of string          |

## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.ReleasePartitions(ctx, "TestCollection", []string{"Partitions"})
// handles the error not nil
```
