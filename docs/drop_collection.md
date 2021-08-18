# Drop Collection

API to drop collection by name.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `name`       | Collection name to drop                                      | String                   |

## Response

`err`: error in the process (if any), possible errors are listed below:

  - `ErrClientNotReady`, error that the client is not connected.

  - `ErrCollectionNotExists`, error that the collection with the specified name does not exist.

  - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance, assume test collection exists
err := cli.DropCollection(ctx, "TestCollection")
// handles the error not nil
```
