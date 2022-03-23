# ListCollection()

This method lists all collections in the connected Milvus instance.

## Invocation

```go
client.ListCollection(ctx)
```

## Parameters
| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process                    | context.Context          |

## Return

A array of entity.Collection that represents the collections in the Milvus instance.

## Errors

`err`: error in the listing process (if any). Possible errors are listed below:

  - `ErrClientNotReady`: error that the client is not connected.

  - Error that API invocation failed.

## Example

```go
listColl, err := milvusClient.ListCollection(
  context.Background(),
)
if err != nil {
  log.Fatal("failed to list all collections:", err.Error())
}
log.Println(listColl)
```
