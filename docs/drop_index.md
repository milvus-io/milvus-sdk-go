# Drop Index 

API to drop index from vector field of a collection.

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to drop index from

- `fieldName` string, the field name to drop index from

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to drop index from                    | String                   |
| `fieldName`  | Name of the field to drop index from                         | String                   |


## Response

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that the specified field is not valid.
    
  - error that API invocation failed.


## Example

```go
ctx := context.Background()
// cli is a valid Client instance
err := cli.DropIndex(ctx, "TestCollection", "Vector")
// handles the error not nil
```
