# Get Index Building Progress 

API to describe index building progress.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to describe index building progress.   | String                   |
| `fieldName`  | Name of the field to describe index building progress.        | String                   |


## Response

- `total`: INT64 number that describes the total records count to build index on.

- `index`: INT64 number that describe the indexed records count.

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that collection with the specified name does not exist.

    - error that the specified field is not valid.
    
    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
indexes, err := cli.GetIndexBuildProgress(ctx, "TestCollection", "Vector")
// handles the error not nil
```
