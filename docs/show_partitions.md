# Show Partitions 

API to list all the partitions in the specified collections 

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, collection name to show;

## Response

- `partitions` array of entity.Partition, represents the partitions of the collection; 

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, error stands for the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
parts, err := cli.ShowPartitions(ctx, "TestCollection")
// handles the error not nil
for _, part := range paritions {
// process each partition if needed
}
```
