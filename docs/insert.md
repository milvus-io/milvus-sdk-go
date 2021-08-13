# Insert 

API to index data into collection 

## Params

- `ctx` context.Context, context to control API invocation process;

- `collName` string, the collection name to insert into;

- `partitionName` string, the partition name to insert into, if empty, default partition will be used;

- `columns` variadic slice of entity.Column, columnar data to insert 

## Response

- `ids` entity.Column, the inserted ids of the data

- `err` error of the creation process (if any), possible error list:

    - ErrClientNotReady, is the client is not connected

    - ErrCollectionNotExists, error stands for collection of the specified name does not exist

    - error for field specified is not valid 
    
    - error fo API invocation failed 

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
// row-base covert to column-base
ids := make([]int64, 0, len(films))
years := make([]int32, 0, len(films))
vectors := make([][]float32, 0, len(films))
// string field is not supported yet
idTitle := make(map[int64]string)
for idx, film := range films {
	ids = append(ids, film.ID)
	idTitle[film.ID] = film.Title
	years = append(years, film.Year)
	vectors = append(vectors, films[idx].Vector[:]) // prevent same vector
}
idColumn := entity.NewColumnInt64("ID", ids)
yearColumn := entity.NewColumnInt32("Year", years)
vectorColumn := entity.NewColumnFloatVector("Vector", 8, vectors)

// insert into default partition
_, err = c.Insert(ctx, collectionName, "", idColumn, yearColumn, vectorColumn)
handles the error not nil
```
