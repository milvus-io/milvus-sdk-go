# Create Index 

API to create index on vector field of a collection.

## Parameters

| Parameter    | Description                                                   | Type                     |
| ------------ | ------------------------------------------------------------- | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to build index on.                     | String                   |
| `fieldName`  | Name of the field to build index on.                          | String                   |
| `idx`        | Index type and specific parameters.                           | entity.Index             |
| `async`      | Switch value of the sync/async behavior. </br>Note: the deadline of context is not applied in sync creation precess. | Boolean |




## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - error that the specified field is not valid.
    
  - error that API invocation failed.

## Example

```go
ctx := context.Background()
idx, err := entity.NewIndexIvfFlat(entity.L2, 2)
// handles err if the newly created index is not valid
// cli is a valid Client instance
err := cli.CreateIndex(ctx, "TestCollection", "Vector", idx, false)
// handles the error not nil
```
