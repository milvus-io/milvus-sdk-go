# Create Collection

API to create collection according to the specified schema.

## Params

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |
| `collSchema` | Collection schema definition                                 | Pointer of entity.Schema |
| `shardNum`   | Shard number of the collection to create. If the `shardNum` is set to 0, default shard number, i.e. 2, will be used. | INT32   |

## Response

`err`: error in the creation process (if any). Possible errors are listed below:

  - `ErrClientNotReady`: error that the client is not connected.

  - error that collection with same name already exists.
    
  - error that API invocation failed.

## Example

```go
ctx := context.Background()
schema := &entity.Schema{
// omit for simpliciy
} 

// cli is a valid Client instance
err := cli.CreateCollection(ctx, schema, 1)
// handles the error not nil
```
