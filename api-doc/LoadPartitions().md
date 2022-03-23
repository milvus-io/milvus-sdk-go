# LoadPartitions()

This method loads partition(s) of the specified collection to memory (for search or query).

## Invocation

```go
client.LoadPartitions(ctx, collName, partNames, async)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to load partition from                | String                   |
| `partNames`  | List of names of the partitions to load                      | Slice of string          |
| `async`      | Switch value to enable the async load. <br/>Note: the deadline of context is not applied in sync load. | Boolean |

## Constraints

- Error will be returned at the attempt to load partition(s) when the parent collection is already loaded. Future releases will support releasing partitions from a loaded collection, and (if needed) then loading some other partition(s).
- "Load successfully" will be returned at the attempt to load the collection that is already loaded.
- Error will be returned at the attempt to load the collection when the child partition(s) is/are already loaded. Future releases will support loading the collection when some of its partitions are already loaded.
- Loading different partitions in a same collection via separate RPCs is not allowed.

## Return

No return.

## Errors

`err`: error in the loading process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
err := milvusClient.LoadPartitions(
  context.Background(),
  "book",
  []string{"novel"},
  false
)
if err != nil {
  log.Fatal("failed to load partitions:", err.Error())
}
```
