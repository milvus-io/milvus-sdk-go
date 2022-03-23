# HasPartition()

This method verifies if a partition exists in the specified collection.

## Invocation

```go
client.HasPartition(ctx, collName, partName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection with the partition to check           | String                   |
| `partName`   | Name of the partition to check                               | String                   |


## Return

Boolean value that stands for whether the partition exists or not.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - error that API invocation failed.

## Example

```go
hasPar, err := milvusClient.HasPartition(
  context.Background(),
  "book",
  "novel",
)
if err != nil {
  log.Fatal("failed to check the partition:", err.Error())
}
log.Println(hasPar)
```
