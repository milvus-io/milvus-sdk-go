# Create Collection

API to create collection according to the specified schema.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collSchema` | Collection schema definition                                 | Pointer of entity.Schema |
| `shardNum`   | Shard number of the collection to create. If the `shardNum` is set to 0, default shard number, i.e. 2, will be used. | INT32   |


## Schema Structure

The schema represents the schema info of the collection in milvus , the schema structure is shown below

| Parameter |  Type |   Description |
| --------- | ------ | ---------- |
| `Collection Name` | string | Name given to the collection |
| `Description` | string | Description given to collection |
| `AutoID` | boolean | The auto id of the collection |
| `Fields` | [Field Schema of milvus](https://github.com/milvus-io/milvus-sdk-go/blob/7410632233597d4af58df727682ffb29f1d1d51d/entity/schema.go#L54-L63) | Contains various fields for the schema  |

## Response

`err`: error in the creation process (if any). Possible errors are listed below:

  - `ErrClientNotReady`: error that the client is not connected.

  - error that collection with same name already exists.
    
  - error that API invocation failed.

## Example

```go
ctx := context.Background()
schema := &entity.Schema{
// omit for simpliciy
} 

// cli is a valid Client instance
err := cli.CreateCollection(ctx, schema, 1)
// handles the error not nil
```
