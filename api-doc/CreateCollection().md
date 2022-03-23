# CreateCollection()

This method creates a collection with the specified schema.

## Invocation

```go
client.CreateCollection(ctx, collSchema, shardNum)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collSchema` | Schema of the collection to create                           | Pointer of entity.Schema |
| `shardNum`   | Shard number of the collection to create. <br/>Default value `2` will be used if it is set as `0`. | INT32   |


A schema specifies the properties of a collection and the fields within. See [Schema](https://milvus.io/docs/v2.0.x/schema.md) for more information.

### Collection schema

A collection schema is the logical definition of a collection.

| Parameter         | Description                                            | Type       |
| ----------------- | ------------------------------------------------------ | ---------- |
| `Collection Name` | Name of the collection to create                       | String     |
| `Description`     | Description of the collection to create                | String     |
| `AutoID`          | Switch value to automatically assigns IDs to entities  | Boolean    |
| `Fields`          | Defines the fields in the collection                   | See [Field Schema of Milvus](https://github.com/milvus-io/milvus-sdk-go/blob/7410632233597d4af58df727682ffb29f1d1d51d/entity/schema.go#L54-L63) for more information. |

### Field schema

A field schema is the logical definition of a field.

| Parameter     |   Description                                           |  Type                          |
| ------------- | ------------------------------------------------------- | ------------------------------ |
|  `ID`         | Field ID generated when collection is created           | INT64                          |
| `Name`        | Name of the field                                       | INT64                          |
| `PrimaryKey`  | Switch value to enable primary key                      | Boolean                        |
| `AutoID`      | Switch value to automatically assigns IDs to entities   | Boolean                        |
| `Description` | Description of the field                                | String                         |
| `DataType`    | Data type of the field.                                 | See [FieldType](https://github.com/milvus-io/milvus-sdk-go/blob/9a7ab65299b4281cc24ad9da7834f6e25866f435/entity/schema.go#L116) for more information.   |
| `TypeParams`  | Type parameters for the field.                          | Map of key string value string |
| `IndexParams` | Index parameters for the field.                         | Map of key string value string |


## Return

A new collection object created with the specified schema.

## Errors

`err`: error in the creation process (if any). Possible errors are listed below:

  - `ErrClientNotReady`: error that the client is not connected.

  - error that collection with same name already exists.
    
  - error that API invocation failed.

## Example

```go
var (
		collectionName = "book"
	)
schema := &entity.Schema{
  CollectionName: collectionName,
  Description:    "Test book search",
  Fields: []*entity.Field{
    {
      Name:       "book_id",
      DataType:   entity.FieldTypeInt64,
      PrimaryKey: true,
      AutoID:     false,
    },
    {
      Name:       "word_count",
      DataType:   entity.FieldTypeInt64,
      PrimaryKey: false,
      AutoID:     false,
    },
    {
      Name:     "book_intro",
      DataType: entity.FieldTypeFloatVector,
      TypeParams: map[string]string{
          "dim": "2",
      },
    },
  },
}
err = milvusClient.CreateCollection(
    context.Background(),
    schema,
    2,
)
if err != nil {
    log.Fatal("failed to create collection:", err.Error())
}
```
