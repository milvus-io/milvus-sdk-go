# Create Collection

API to create a collection according to the specified schema.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collSchema` | Collection schema definition                                 | Pointer of entity.Schema |
| `shardNum`   | Shard number of the collection to create. If the `shardNum` is set to 0, default shard number will be used. | INT32   |


## Schema

A schema specifies the features of the collection to create and the fields within the collection.

### Collection schema

A collection schema is the logical definition of a collection.

| Parameter |   Description |  Type |
| --------- | ------ | ---------- |
| `Collection Name` | Name of the collection to create. | String |
| `Description` | Description of the collection to create. | String |
| `AutoID` | Automatically assigns IDs to entities in the collection if it is set to `true`. | Boolean |
| `Fields` | Defines the fields in the collection.  | See [Field Schema of Milvus](https://github.com/milvus-io/milvus-sdk-go/blob/7410632233597d4af58df727682ffb29f1d1d51d/entity/schema.go#L54-L63) for more information. |

### Field schema

A field schema is the logical definition of a field.

| Parameter  |   Description                                  |  Type        |
| ---------- | ---------------------------------------------- | ------------ |
|  `ID`      | Field ID generated when collection is created. | int64        |
| `Name`     | Name of the field.                             | int64        |
| `PrimaryKey` | Switch value of primary key enablement.      | Boolean      |
| `AutoID`   | Switch value of auto-generated ID enablement.  | Boolean |
| `Description` | Description of the field.                   | String       |
| `DataType` | Data type of the field. | See [FieldType](https://github.com/milvus-io/milvus-sdk-go/blob/9a7ab65299b4281cc24ad9da7834f6e25866f435/entity/schema.go#L116) for more information. |
| `TypeParams` | Type parameters for the field.               | Map of key string value string |
| `IndexParams` | Index parameters for the field.             | Map of key string value string |


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
