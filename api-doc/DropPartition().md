# DropPartition()

This method drops the specified partition and the entities within.

## Invocation

```go
client.DropPartition(ctx, collName, partName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to drop partition from                | String                   |
| `partName`   | Name of the partition to drop                                | String                   |

## Return

No return.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error stands for the client is not connected

  - `ErrCollectionNotExists`, error stands for collection of the specified name does not exist

  - error that API invocation failed.

## Example

```go
err := milvusClient.DropPartition(
  context.Background(),
  "book",
  "novel",
)
if err != nil {
  log.Fatal("fail to drop partition:", err.Error())
}
```
