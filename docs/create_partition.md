# Create Partition 

API to create partition in a collection 

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name;

- `partitionName` string, the partition name to create; 

## Response

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error for partition with same name already exists
    
    - error fo API invocation failed 

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
