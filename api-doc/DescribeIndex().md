# DescribeIndex()

This method gets the detailed information of an index.

## Invocation

```go
client.DescribeIndex(ctx, collName, fieldName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to describe index on                  | String                   |
| `fieldName`  | Name of the field on which to describe the index             | String                   |

## Return

A slice of entity.Index that specifies the indexes on the vector field.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that the specified field is not valid.
    
  - error that API invocation failed.

## Example

```go
indexInfo, err := milvusClient.DescribeIndex(
  context.Background(),
  "book",
  "book_intro"
)
if err != nil {
  log.Fatal("fail to describe index:", err.Error())
}
log.Println(indexInfo)
```
