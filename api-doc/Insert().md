# Insert()

This method inserts data into a specified collection.

## Invocation

```go
client.Insert(ctx, collName, partName, data)
```

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collName`   | Name of the collection to insert data into                   | String                   |
| `partName`   | Name of the partition to insert data into. <br/>Data will be inserted into default partition if it is left empty. | String |
| `data`       | Data to insert in each field. <br/>Data must be converted into columns for different fields. | Variadic slice of entity.Column |



## Return

Entity.Column that represents the IDs of the inserted data.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that the specified field is not valid.
    
  - error that API invocation failed.

## Example

```go
bookIDs := make([]int64, 0, 2000)
wordCounts := make([]int64, 0, 2000)
bookIntros := make([][]float32, 0, 2000)
for i := 0; i < 2000; i++ {
	bookIDs = append(bookIDs, int64(i))
	wordCounts = append(wordCounts, int64(i+10000))
	v := make([]float32, 0, 2)
	for j := 0; j < 2; j++ {
		v = append(v, rand.Float32())
	}
	bookIntros = append(bookIntros, v)
}
idColumn := entity.NewColumnInt64("book_id", bookIDs)
wordColumn := entity.NewColumnInt64("word_count", wordCounts)
introColumn := entity.NewColumnFloatVector("book_intro", 2, bookIntros)
_, err = milvusClient.Insert(
	context.Background(),
	"book",
	"",
	idColumn,
	wordColumn,
	introColumn,
)
if err != nil {
	log.Fatal("failed to insert data:", err.Error())
}
```
