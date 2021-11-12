# Describe Collection

API to get collection information.

## Parameters

| Parameter    | Description                                                   | Type                     |
| ------------ | ------------------------------------------------------------- | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to describe.                           | String                   |




## Response

- `collection`: pointer of entity.Collection that represents the collection to describe.

- `err`: error in the process (if any), possible errors are listed:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
coll, err := cli.DescribeCollection(ctx, "TestCollection")
// handles the error not nil
```
