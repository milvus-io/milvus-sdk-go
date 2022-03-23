# ReleasePartitions()

This method releases partition(s) of the specified collection from memory.

## Invocation

```go
client.ReleasePartition(ctx, collName, partNames)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to release partitions from            | String                   |
| `partNames`  | List of names of the partitions to release                   | Slice of string          |

## Constraints

- Error will be returned at the attempt to load partition(s) when the parent collection is already loaded. Future releases will support releasing partitions from a loaded collection, and (if needed) then loading some other partition(s).
- "Load successfully" will be returned at the attempt to load the collection that is already loaded.
- Error will be returned at the attempt to load the collection when the child partition(s) is/are already loaded. Future releases will support loading the collection when some of its partitions are already loaded.
- Loading different partitions in a same collection via separate RPCs is not allowed.

## Return

No return.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
err := milvusClient.ReleasePartitions(
  context.Background(),
  "book",
  []string{"novel"}
)
if err != nil {
  log.Fatal("failed to release partitions:", err.Error())
}
```
