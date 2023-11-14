package client

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
)

func WithCreateCollectionMsgBase(msgBase *commonpb.MsgBase) CreateCollectionOption {
	return func(opt *createCollOpt) {
		opt.MsgBase = msgBase
	}
}

func WithDropCollectionMsgBase(msgBase *commonpb.MsgBase) DropCollectionOption {
	return func(req *milvuspb.DropCollectionRequest) {
		req.Base = msgBase
	}
}

func WithLoadCollectionMsgBase(msgBase *commonpb.MsgBase) LoadCollectionOption {
	return func(req *milvuspb.LoadCollectionRequest) {
		req.Base = msgBase
	}
}

func WithReleaseCollectionMsgBase(msgBase *commonpb.MsgBase) ReleaseCollectionOption {
	return func(req *milvuspb.ReleaseCollectionRequest) {
		req.Base = msgBase
	}
}

func WithFlushMsgBase(msgBase *commonpb.MsgBase) FlushOption {
	return func(req *milvuspb.FlushRequest) {
		req.Base = msgBase
	}
}

func WithCreateDatabaseMsgBase(msgBase *commonpb.MsgBase) CreateDatabaseOption {
	return func(req *milvuspb.CreateDatabaseRequest) {
		req.Base = msgBase
	}
}

func WithDropDatabaseMsgBase(msgBase *commonpb.MsgBase) DropDatabaseOption {
	return func(req *milvuspb.DropDatabaseRequest) {
		req.Base = msgBase
	}
}

func WithReplicateMessageMsgBase(msgBase *commonpb.MsgBase) ReplicateMessageOption {
	return func(req *milvuspb.ReplicateMessageRequest) {
		req.Base = msgBase
	}
}

func WithCreatePartitionMsgBase(msgBase *commonpb.MsgBase) CreatePartitionOption {
	return func(req *milvuspb.CreatePartitionRequest) {
		req.Base = msgBase
	}
}

func WithDropPartitionMsgBase(msgBase *commonpb.MsgBase) DropPartitionOption {
	return func(req *milvuspb.DropPartitionRequest) {
		req.Base = msgBase
	}
}

func WithLoadPartitionsMsgBase(msgBase *commonpb.MsgBase) LoadPartitionsOption {
	return func(req *milvuspb.LoadPartitionsRequest) {
		req.Base = msgBase
	}
}

// Deprecated: use WithReleaseCollectionMsgBase instead
func WithReleasePartitionMsgBase(msgBase *commonpb.MsgBase) ReleasePartitionsOption {
	return WithReleasePartitionsMsgBase(msgBase)
}

func WithReleasePartitionsMsgBase(msgBase *commonpb.MsgBase) ReleasePartitionsOption {
	return func(req *milvuspb.ReleasePartitionsRequest) {
		req.Base = msgBase
	}
}
