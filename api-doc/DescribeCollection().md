# DescribeCollection()

This method gets the detailed information of the specified collection.

## Invocation

```go
client.DescribeCollection(stc, collName)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to describe                           | String                   |

## Return

Pointer of entity.Collection that represents the collection to describe.

## Errors

- `err`: error in the process (if any), possible errors are listed:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

    - error that API invocation failed.

## Example

```go
collDesc, err := milvusClient.DescribeCollection(
  context.Background(),
  "book",
)
if err != nil {
  log.Fatal("failed to check collection schema:", err.Error())
}
log.Printf("%v\n", collDesc)
```
