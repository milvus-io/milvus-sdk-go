# Show Partitions 

API to list all the partitions in the specified collection.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to list all partitions.                | String                   |


## Response

- `partitions`: array of entity.Partition that represents the partitions in the collection.

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
partitions, err := cli.ShowPartitions(ctx, "TestCollection")
// handles the error not nil
for _, part := range partitions {
// process each partition if needed
}
```
