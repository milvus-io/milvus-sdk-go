# CreateIndex()

This method creates an index for the specified vector field.

## Invocation

```go
client.CreateIndex(ctx, collName, fieldName, idx, async)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to build index on                     | String                   |
| `fieldName`  | Name of the vector field to build index on                   | String                   |
| `idx`        | Index type and specific parameters                           | entity.Index             |
| `async`      | Switch value to enable async index building. </br>Note: the deadline of context is not applied in sync creation precess. | Boolean |


## Return

No return.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - error that the specified field is not valid.
    
  - error that API invocation failed.

## Example

```go
idx, err := entity.NewIndexIvfFlat(
    entity.L2,
    1024,
)
if err != nil {
  log.Fatal("fail to create ivf flat index parameter:", err.Error())
}
err = milvusClient.CreateIndex(
  context.Background(),
  "book",
  "book_intro",
  idx,
  false,
)
if err != nil {
  log.Fatal("fail to create index:", err.Error())
}
```
