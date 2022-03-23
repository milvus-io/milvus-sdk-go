# CreatePartition()

This method creates a partition in the specified collection.

## Invocation

```go
client.CreatePartition(ctx, collName, partName)
```

## Parameters

| Parameter    | Description                                                   | Type                     |
| ------------ | ------------------------------------------------------------- | ------------------------ |
| `ctx`        | Context to control API invocation process                     | context.Context          |
| `collName`   | Name of the collection to create a partition in               | String                   |
| `partName`   | Name of the partition to create                               | String                   |

## Return

A new partition object corresponded to the name in the specified collection.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that partition with the same name already exists.

  - error that API invocation failed.

## Example

```go
err := milvusClient.CreatePartition(
  context.Background(),
  "book",
  "novel"
)
if err != nil {
  log.Fatal("failed to create partition:", err.Error())
}
```
