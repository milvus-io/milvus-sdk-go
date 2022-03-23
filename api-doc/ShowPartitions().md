# ShowPartitions()

This method lists all partitions in the specified collection.

## Invocation

```go
client.ShowPartitions(ctx, collName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection in which to list all partitions       | String                   |


## Return

An array of entity.Partition that represents the partitions in the collection.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
listPar, err := milvusClient.ShowPartitions(
  context.Background(),
  "book",
)
if err != nil {
  log.Fatal("failed to list partitions:", err.Error())
}
log.Println(listPar)
```
