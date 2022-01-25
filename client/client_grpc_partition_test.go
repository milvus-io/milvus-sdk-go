package client

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/common"
	"github.com/milvus-io/milvus-sdk-go/v2/internal/proto/server"
	"github.com/stretchr/testify/assert"
)

// return injection asserts collection name matchs
// partition name request in partitionNames if flag is true
func hasPartitionInjection(t *testing.T, collName string, mustIn bool, partitionNames ...string) func(context.Context, proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.HasPartitionRequest)
		resp := &server.BoolResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, collName, req.GetCollectionName())
		if mustIn {
			resp.Value = assert.Contains(t, partitionNames, req.GetPartitionName())
		} else {
			for _, pn := range partitionNames {
				if pn == req.GetPartitionName() {
					resp.Value = true
				}
			}
		}
		s, err := successStatus()
		resp.Status = s
		return resp, err
	}
}

func TestGrpcClientCreatePartition(t *testing.T) {

	ctx := context.Background()
	c := testClient(ctx, t)

	partitionName := fmt.Sprintf("_part_%d", rand.Int())

	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mHasPartition, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*server.HasPartitionRequest)
		resp := &server.BoolResponse{}
		if !ok {
			s, err := badRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.Equal(t, partitionName, req.GetPartitionName())
		resp.Value = false
		s, err := successStatus()
		resp.Status = s
		return resp, err
	})

	assert.Nil(t, c.CreatePartition(ctx, testCollectionName, partitionName))
}

func TestGrpcClientDropPartition(t *testing.T) {
	partitionName := fmt.Sprintf("_part_%d", rand.Int())
	ctx := context.Background()
	c := testClient(ctx, t)
	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, true, partitionName)) // injection has assertion of collName & parition name
	assert.Nil(t, c.DropPartition(ctx, testCollectionName, partitionName))
}

func TestGrpcClientHasPartition(t *testing.T) {
	partitionName := fmt.Sprintf("_part_%d", rand.Int())
	ctx := context.Background()
	c := testClient(ctx, t)
	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, false, partitionName)) // injection has assertion of collName & parition name

	r, err := c.HasPartition(ctx, testCollectionName, "_default_part")
	assert.Nil(t, err)
	assert.False(t, r)

	r, err = c.HasPartition(ctx, testCollectionName, partitionName)
	assert.Nil(t, err)
	assert.True(t, r)
}

// default partition interception for ShowPartitions, generates testCollection related paritition data
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
			resp.InMemoryPercentages = append(resp.InMemoryPercentages, 100)
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

	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, true, "_part1", "_part2", "_part3", "_part4"))

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
		start := time.Now()
		loadTime := rand.Intn(1500) + 100
		loaded := false
		mock.setInjection(mShowPartitions, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.ShowPartitionsRequest)
			resp := &server.ShowPartitionsResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			resp.PartitionIDs = make([]int64, 0, len(tc.loadNames))
			resp.PartitionNames = make([]string, 0, len(tc.loadNames))
			var perc int64 = 0
			if time.Since(start) > time.Duration(loadTime*int(time.Millisecond)) {
				perc = 100
				loaded = true
			}
			for _, part := range tc.partitions {
				resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
				resp.PartitionNames = append(resp.PartitionNames, part.Name)
				resp.InMemoryPercentages = append(resp.InMemoryPercentages, perc)
			}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})

		err := c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		if tc.shouldSuccess {
			assert.Nil(t, err)
			assert.True(t, loaded)
		} else {
			assert.NotNil(t, err)
		}
	}

	t.Run("some service failed", func(t *testing.T) {
		tc := &testCase{
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
			loadNames: []string{"_part1", "_part2"},
		}

		// paritions will not be loaded
		mock.setInjection(mShowPartitions, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			req, ok := raw.(*server.ShowPartitionsRequest)
			resp := &server.ShowPartitionsResponse{}
			if !ok {
				s, err := badRequestStatus()
				resp.Status = s
				return resp, err
			}
			assert.Equal(t, testCollectionName, req.GetCollectionName())
			resp.PartitionIDs = make([]int64, 0, len(tc.loadNames))
			resp.PartitionNames = make([]string, 0, len(tc.loadNames))
			var perc int64 = 0
			for _, part := range tc.partitions {
				resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
				resp.PartitionNames = append(resp.PartitionNames, part.Name)
				resp.InMemoryPercentages = append(resp.InMemoryPercentages, perc)
			}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		quickCtx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
		defer cancel()
		assert.NotNil(t, c.LoadPartitions(quickCtx, tc.collName, tc.loadNames, false))

		mock.setInjection(mShowPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &server.ShowPartitionsResponse{}, errors.New("always fail")
		})
		defer mock.delInjection(mShowPartitions)

		err := c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		assert.NotNil(t, err)

		mock.setInjection(mLoadPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &common.Status{}, errors.New("has partition failed")
		})
		err = c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		assert.NotNil(t, err)

		mock.setInjection(mHasPartition, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &server.BoolResponse{}, errors.New("has partition failed")
		})
		err = c.LoadPartitions(ctx, tc.collName, tc.loadNames, false)
		assert.NotNil(t, err)

	})
}

func TestGrpcClientReleasePartitions(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	parts := []string{"_part1", "_part2"}
	mock.setInjection(mHasCollection, hasCollectionDefault)
	mock.setInjection(mHasPartition, hasPartitionInjection(t, testCollectionName, true, "_part1", "_part2", "_part3", "_part4"))
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

func TestGrpcShowPartitions(t *testing.T) {
	ctx := context.Background()
	c := testClient(ctx, t)

	partitions := []*entity.Partition{
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
	}

	t.Run("normal show partitions", func(t *testing.T) {
		mock.setInjection(mShowPartitions, getPartitionsInterception(t, testCollectionName, partitions...))
		parts, err := c.ShowPartitions(ctx, testCollectionName)
		assert.NoError(t, err)
		assert.NotNil(t, parts)
	})

	t.Run("bad response", func(t *testing.T) {
		mock.setInjection(mShowPartitions, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &server.ShowPartitionsResponse{}
			resp.PartitionIDs = make([]int64, 0, len(partitions))
			for _, part := range partitions {
				resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
			}
			s, err := successStatus()
			resp.Status = s
			return resp, err
		})
		_, err := c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)
	})

	t.Run("service error", func(t *testing.T) {
		mock.setInjection(mShowPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &server.ShowPartitionsResponse{}, errors.New("always fail")
		})
		defer mock.delInjection(mShowPartitions)

		_, err := c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)

		mock.setInjection(mShowPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &server.ShowPartitionsResponse{
				Status: &common.Status{ErrorCode: common.ErrorCode_UnexpectedError},
			}, nil
		})
		_, err = c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)
	})
}
