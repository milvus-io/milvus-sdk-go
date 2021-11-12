# Search 

API to search data with the specified criteria.

## Params

| Parameter    | Description                                | Type            |
| ------------ | ------------------------------------------ | --------------- |
| `ctx`        | Context to control API invocation process. | context.Context |
| `collName`   | Name of the collection to search in.       | String          |
| `partitions` | List of names of the partitions to search in. </br>If empty, all partition will be searched. | Slice of string |
| `expr`       | Boolean expression to filter the data.     | String          |
| `outputFields` | List of names of fields to output.       | Slice of string |
| `vectors`    | Vectors to search with.                    | Slice of entity.Vector |
| `metricType` | Metric type to calculate distance with.    | entity.MetricType |
| `topK`       | Number of nearest records to return.       | INT             |
| `sp`         | Specific search parameter(s) that related to the index type the vector field has. | entity.SearchParam |



## Response

- `results`: slice of SearchResult that contains the search result, one record per vector.

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// vector is the vector to search
// Use flat search param
sp, _ := entity.NewIndexFlatSearchParam(10)
// cli is a valid Client instance
sr, err := c.Search(ctx, collectionName, []string{}, "Year > 1990", []string{"ID"}, []entity.Vector{vector}, "Vector",
	entity.L2, 10, sp)
// some example code to process SearchResult
for _, result := range sr {
	var idColumn *entity.ColumnInt64
	for _, field := range result.Fields {
		if field.Name() == "ID" {
			c, ok := field.(*entity.ColumnInt64)
			if ok {
				idColumn = c
			}
		}
	}
	if idColumn == nil {
		log.Fatal("result field not math")
	}
	for i := 0; i < result.ResultCount; i++ {
		id, err := idColumn.ValueByIdx(i)
		if err != nil {
			log.Fatal(err.Error())
		}
		title := idTitle[id]
		fmt.Printf("file id: %d title: %s scores: %f\n", id, title, result.Scores[i])
	}
}

```
