# Flush 

API to flush the newly inserted data inserted.

## Parameters

| Parameter    | Description                                                  | Type                     |
| ------------ | ------------------------------------------------------------ | ------------------------ |
| `ctx`        | Context to control API invocation process.                    | context.Context          |
| `collName`   | Name of the collection to flush.                              | String                   |
| `async`      | Switch value of the sync/async behavior. </br>Note: the deadline of context is not applied in sync flush. | Boolean |

## Response

- `err`: error in the process (if any). Possible errors are listed below:

    - `ErrClientNotReady`, error that the client is not connected.

    - `ErrCollectionNotExists`, error that collection with the specified name does not exist.
    
    - error that API invocation failed.

## Example

```go
ctx := context.Background()
// cli is a valid Client instance
err := c.Flush(ctx, "TestCollection", false)
handles the error not nil
```
