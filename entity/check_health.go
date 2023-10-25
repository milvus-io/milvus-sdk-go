package entity

type QuotaState int32

const (
	// QuotaStateUnknown zero value placeholder
	QuotaStateUnknown QuotaState = 0
	// QuotaStateReadLimited too many read tasks, read requests are limited
	QuotaStateReadLimited QuotaState = 2
	// QuotaStateWriteLimited too many write tasks, write requests are limited
	QuotaStateWriteLimited QuotaState = 3
	// QuotaStateDenyToRead too many read tasks, temporarily unable to process read requests
	QuotaStateDenyToRead QuotaState = 4
	// QuotaStateDenyToWrite too many write tasks, temporarily unable to process write requests
	QuotaStateDenyToWrite QuotaState = 5
)

type MilvusState struct {
	IsHealthy   bool
	Reasons     []string
	QuotaStates []QuotaState
}
