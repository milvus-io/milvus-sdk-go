package client

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

func TestGrpcClientCreatePartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	assert.Nil(t, c.CreatePartition(ctx, testCollectionName, "_default_part"))
}

func TestGrpcClientDropPartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	assert.Nil(t, c.DropPartition(ctx, testCollectionName, "_default_part"))
}

func TestGrpcClientHasPartition(t *testing.T) {
	//TODO add detail asserts
	ctx := context.Background()
	c := testClient(ctx, t)
	r, err := c.HasPartition(ctx, testCollectionName, "_default_part")
	assert.Nil(t, err)
	assert.False(t, r)
}

// default partition intercetion for ShowPartitions, generates testCollection related paritition data
func getPartitionsInterception(t *testing.T, collName string, partitions ...*entity.Partition) func(ctx context.Context, raw proto.Message) (proto.Message, error) {
	return func(ctx context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ShowPartitionsRequest)
		resp := &server.ShowPartitionsResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, collName, req.GetCollectionName())
		resp.PartitionIDs = make([]int64, 0, len(partitions))
		resp.PartitionNames = make([]string, 0, len(partitions))
		for _, part := range partitions {
			resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
			resp.PartitionNames = append(resp.PartitionNames, part.Name)
		}
		s, err := successStatus()
		resp.Status = s
		return resp, err
	}
}

func TestGrpcClientShowPartitions(t *testing.T) {

	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		collName      string
		partitions    []*entity.Partition
		shouldSuccess bool
	}
	cases := []testCase{
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			shouldSuccess: true,
		},
	}
	for _, tc := range cases {
		mock.setInjection(mShowPartitions, getPartitionsInterception(t, tc.collName, tc.partitions...))
		r, err := c.ShowPartitions(ctx, tc.collName)
		if tc.shouldSuccess {
			assert.Nil(t, err)
			assert.NotNil(t, r)
			if assert.Equal(t, len(tc.partitions), len(r)) {
				for idx, part := range tc.partitions {
					assert.Equal(t, part.ID, r[idx].ID)
					assert.Equal(t, part.Name, r[idx].Name)
				}
			}
		} else {
			assert.NotNil(t, err)
		}
	}
}

func TestGrpcClientLoadPartitions(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	type testCase struct {
		collName      string
		partitions    []*entity.Partition
		shouldSuccess bool
		loadNames     []string
	}
	cases := []testCase{
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			loadNames:     []string{"_part1", "_part2"},
			shouldSuccess: true,
		},
		{
			collName: testCollectionName,
			partitions: []*entity.Partition{
				{
					ID:   1,
					Name: "_part1",
				},
				{
					ID:   2,
					Name: "_part2",
				},
				{
					ID:   3,
					Name: "_part3",
				},
			},
			loadNames:     []string{"_part4", "_part1"},
			shouldSuccess: false,
		},
	}
	for _, tc := range cases {
		// one segment per paritions
		segmentCount := len(tc.partitions)
		start := time.Now()
		loadTime := rand.Intn(1500) + 100
		rowCounts := rand.Intn(1000) + 100
		ok := false
		mock.setInjection(mGetPersistentSegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetPersistentSegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.PersistentSegmentInfo, 0, segmentCount),
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
					SegmentID:   int64(i),
					NumRows:     int64(rowCounts),
					PartitionID: tc.partitions[i].ID,
				})
			}
			r.Infos = append(r.Infos, &server.PersistentSegmentInfo{
				SegmentID: int64(segmentCount),
				NumRows:   0, // handcrafted empty segment
			})
			return r, err
		})
		mock.setInjection(mGetQuerySegmentInfo, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			s, err := successStatus()
			r := &server.GetQuerySegmentInfoResponse{
				Status: s,
				Infos:  make([]*server.QuerySegmentInfo, 0, segmentCount),
			}
			rc := 0
			if time.Since(start) > time.Duration(loadTime)*time.Millisecond {
				rc = rowCounts // after load time, row counts set to full amount
				ok = true
			}
			for i := 0; i < segmentCount; i++ {
				r.Infos = append(r.Infos, &server.QuerySegmentInfo{
					SegmentID:   int64(i),
					NumRows:     int64(rc),
					PartitionID: tc.partitions[i].ID,
				})
			}
			return r, err
		})

		mock.setInjection(mShowPartitions, getPartitionsInterception(t, tc.collName, tc.partitions...))
		err := c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		if tc.shouldSuccess {
			assert.Nil(t, err)
			assert.True(t, ok)
		} else {
			assert.NotNil(t, err)
		}
	}
}

func TestGrpcClientReleasePartitions(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	parts := []string{"_part1", "_part2"}

	mock.setInjection(mReleasePartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.ReleasePartitionsRequest)
		if !ok {
			return badRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.ElementsMatch(t, parts, req.GetPartitionNames())

		return successStatus()
	})

	assert.Nil(t, c.ReleasePartitions(ctx, testCollectionName, parts))
}
