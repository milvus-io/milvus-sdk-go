package entity

//ResourceGroup information model struct.
type ResourceGroup struct {
	Name                 string
	Capacity             int32
	AvailableNodesNumber int32
	LoadedReplica        map[string]int32
	OutgoingNodeNum      map[string]int32
	IncomingNodeNum      map[string]int32
}
