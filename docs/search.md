# Search 

API to search data with criteria provided

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to search;

- `partitions` slice of string, the partition names to search with, if empty, all partition will be used;

- `expr` string, the bool expression to filter the data

- `outputFields` slice of string, the output fields

- `vectors` slice of entity.Vector, the vectors to search with

- `metricType` entity.MetricType, the metric type to calculate distance with

- `topK` int, the number of nearest record to return

- `sp` entity.SearchParam, the specified search param, which is related to the index type vector field has

## Response

- `results` slice of SearchResult, one record per vector, contains the search result; 

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error fo API invocation failed 

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
