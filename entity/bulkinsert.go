package entity

import "strconv"

type BulkInsertState int32

const (
	BulkInsertPending          BulkInsertState = 0 // the task in in pending list of rootCoord, waiting to be executed
	BulkInsertFailed           BulkInsertState = 1 // the task failed for some reason, get detail reason from GetImportStateResponse.infos
	BulkInsertStarted          BulkInsertState = 2 // the task has been sent to datanode to execute
	BulkInsertPersisted        BulkInsertState = 5 // all data files have been parsed and data already persisted
	BulkInsertCompleted        BulkInsertState = 6 // all indexes are successfully built and segments are able to be compacted as normal.
	BulkInsertFailedAndCleaned BulkInsertState = 7 // the task failed and all segments it generated are cleaned up.

	ImportProgress = "progress_percent"
)

type BulkInsertTaskState struct {
	ID           int64             // id of an import task
	State        BulkInsertState   // is this import task finished or not
	RowCount     int64             // if the task is finished, this value is how many rows are imported. if the task is not finished, this value is how many rows are parsed. return 0 if failed.
	IDList       []int64           // auto generated ids if the primary key is autoid
	Infos        map[string]string // more information about the task, progress percent, file path, failed reason, etc.
	CollectionID int64             // collection ID of the import task.
	SegmentIDs   []int64           // a list of segment IDs created by the import task.
	CreateTs     int64             //timestamp when the import task is created.
}

func (state BulkInsertTaskState) Progress() int {
	if val, ok := state.Infos[ImportProgress]; ok {
		progress, err := strconv.Atoi(val)
		if err != nil {
			return 0
		}
		return progress
	}
	return 0
}
