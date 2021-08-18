# Create Partition 

API to create partition in a collection.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Collection name to create partition in                       | String                   |
| `partitionName` | Partition name to create                                  | String                   |



## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that partition with the same name already exists.

  - error that API invocation failed.

## Example

```go
ctx := context.Background()
schema := &entity.Schema{
// omit for simpliciy
} 

// cli is a valid Client instance
err := cli.CreatePartition(ctx, "TestCollection", "Partition1")
// handles the error not nil
```
