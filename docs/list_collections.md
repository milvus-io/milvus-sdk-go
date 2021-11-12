# List Collections

API to list all collections in the connected Milvus instance.

## Parameters
| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |

## Response

- `collections`: array of entity.Collection that represents the collections in the Milvus instance.

- `err`: error in the listing process (if any). Possible errors are listed below:

    - `ErrClientNotReady`: error that the client is not connected.

    - Error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
colls, err := cli.ListCollection(ctx)
// handles the error not nil
for _, coll := range colls {
// process each collection if needed
}
```
