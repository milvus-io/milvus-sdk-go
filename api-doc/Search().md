# Search()

This method conducts a vector similarity search.

## Invocation

```go
client.Search(ctx, collName, partNames, expr, outputFields, vectors, fieldName, metricType, topK, sp)
```

## Parameters

| Parameter      | Description                                                   | Type                   |
| -------------- | ------------------------------------------------------------- | ---------------------- |
| `ctx`          | Context to control API invocation process                     | context.Context        |
| `collName`     | Name of the collection to search in                           | String                 |
| `partNames`    | List of names of the partitions to search on. </br>All partition will be searched if it is left empty. | Slice of string |
| `expr`         | Boolean expression to filter the data                         | String                 |
| `outputFields` | List of names of fields to output                             | Slice of string        |
| `vectors`      | Vectors to search with                                        | Slice of entity.Vector |
| `fieldName`    | Name of the vector field to search on                         | String                 |
| `metricType`   | Metric type to calculate distance with                        | entity.MetricType      |
| `topK`         | Number of nearest records to return                           | INT                    |
| `sp`           | Specific search parameter(s) of the index on the vector field | entity.SearchParam     |



## Return

A slice of SearchResult that contains the search result, one record per vector.

## Errors

`err`: error in the process (if any). Possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
searchResult, err := milvusClient.Search(
	context.Background(),                    // ctx
	"book",                                  // CollectionName
	[]string{},                              // partitionNames
	"",                                      // expr
	[]string{"book_id"},                     // outputFields
	[]entity.Vector{entity.FloatVector([]float32{0.1, 0.2})}, // vectors
	"book_intro",                            // vectorField
	entity.L2,                               // metricType
	2,                                       // topK
	sp,                                      // sp
)
if err != nil {
	log.Fatal("fail to search collection:", err.Error())
}
fmt.Printf("%#v\n", searchResult)
for _, sr := range searchResult {
	fmt.Println(sr.IDs)
	fmt.Println(sr.Scores)
}
```
