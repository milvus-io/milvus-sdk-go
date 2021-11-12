# Insert 

API to insert data into a specified collection.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to insert data into.                        | String                   |
| `partitionName` | Name of the collection to insert data into. </br>If empty, default partition will be used. | String |
| `columns`    | Columnar data to insert.                                      | Variadic slice of entity.Column |



## Response

- `ids`: entity.Column that represents the IDs of the inserted data

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

    - error that the specified field is not valid.
    
    - error that API invocation failed.

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
