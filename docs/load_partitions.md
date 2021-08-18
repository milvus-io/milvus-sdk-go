# Load Partitions

API to load partitions by name.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to load partition from                | String                   |
| `partitionNames` | Name of the Partition to load                            | Slice of string          |
| `async`      | Switch value of the sync/async behavior. </br>Note: the deadline of context is not applied in sync load. | Boolean |

## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.LoadPartitions(ctx, "TestCollection",[]string{"Partitions"}, false)
// handles the error not nil
```
