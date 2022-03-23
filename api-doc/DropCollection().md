# DropCollection()

This method drops the specified collection and the entities within.

## Invocation

```go
client.DropCollection(ctx, collName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to drop                               | String                   |

## Return

No return.

## Errors

`err`: error in the dropping process (if any), possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
err = milvusClient.DropCollection(
  context.Background(),
  "book",
)
if err != nil {
	log.Fatal("fail to drop collection:", err.Error())
}
```
