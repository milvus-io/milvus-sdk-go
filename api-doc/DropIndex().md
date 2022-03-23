# DropIndex()

This method drops the specified index and the corresponding index file.

## Invocation

```go
client.DropIndex(ctx, collName, fieldName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to drop index from                    | String                   |
| `fieldName`  | Name of the vector field to drop index from                  | String                   |


## Return

No return.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that the specified field is not valid.
    
  - error that API invocation failed.


## Example

```go
err = milvusClient.DropIndex(
  context.Background(),
  "book",
  "book_intro",
)
if err != nil {
  log.Fatal("fail to drop index:", err.Error())
}
```
