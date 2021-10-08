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

The below schema is for the fields parameter of the above collection schema

| Parameter |  Type |   Description |
| --------- | ------ | ---------- |
|  `ID` | int64 | Field id generated when collection is made |
| `Name` | string | Name of field |
| `PrimaryKey` | bool | Primary Key of the schema |
| `AutoID` | bool | AutoID of the schema |
| `Description` | string | Description of the schema |
| `DataType` | [FieldType](https://github.com/milvus-io/milvus-sdk-go/blob/9a7ab65299b4281cc24ad9da7834f6e25866f435/entity/schema.go#L116) | DataType of field |
| `TypeParams` | map of key string value string | Type Params for the schema |
| `IndexParams` | map of key string value string | Index Params for the schema |


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
