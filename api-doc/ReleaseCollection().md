# ReleaseCollection()

This method releases the specified collection from memory.

## Invocation

```go
client.ReleaseCollection(ctx, collName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to release                            | String                   |

## Constraints

- Releasing the collection that is successfully loaded is allowed.
- Releasing the collection is allowed when its partition(s) are loaded.
- Error will be returned at the attempt to release partition(s) when the parent collection is already loaded. Future releases will support releasing partitions from a loaded collection, and loading the collection when its partition(s) are released.

## Return

No return.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
err := milvusClient.ReleaseCollection(
  context.Background(),
  "book",
)
if err != nil {
  log.Fatal("failed to release collection:", err.Error())
}
```
