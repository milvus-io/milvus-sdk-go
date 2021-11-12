# Describe Index 

API to describe index on vector field of a collection.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to describe index on.                  | String                   |
| `fieldName`  | Name of the field to describe index on.                       | String                   |

## Response

- `indexes`: slice of entity.Index that specifies the indexes on the vector field.

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

    - error that the specified field is not valid.
    
    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
indexes, err := cli.DescribeIndex(ctx, "TestCollection", "Vector")
// handles the error not nil
```
