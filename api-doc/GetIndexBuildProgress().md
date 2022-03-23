# GetIndexBuildProgress()

This method gets the building progress of the specified index.

## Invocation

```go
client.GetIndexBuildProgress(ctx, collName, fieldName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to describe index building progress   | String                   |
| `fieldName`  | Name of the field to describe index building progress        | String                   |


## Return

- `total`: INT64 number that describes the total records count to build index on.

- `index`: INT64 number that describe the indexed records count.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that the specified field is not valid.
    
  - error that API invocation failed.

## Example

```go
indexProgress, err := milvusClient.GetIndexBuildProgress(
  context.Background(),
  "book",
  "book_intro",
)
if err != nil {
  log.Fatal("fail to get index building progress:", err.Error())
}
```
