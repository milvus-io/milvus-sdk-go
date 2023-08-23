package client

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/cockroachdb/errors"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

// return injection asserts collection name matchs
// partition name request in partitionNames if flag is true
func hasPartitionInjection(t *testing.T, collName string, mustIn bool, partitionNames ...string) func(context.Context, proto.Message) (proto.Message, error) {
	return func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.HasPartitionRequest)
		resp := &milvuspb.BoolResponse{}
		if !ok {
			s, err := BadRequestStatus()
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
		s, err := SuccessStatus()
		resp.Status = s
		return resp, err
	}
}

func TestGrpcClientCreatePartition(t *testing.T) {

	ctx := context.Background()
	c := testClient(ctx, t)

	partitionName := fmt.Sprintf("_part_%d", rand.Int())

	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MHasPartition, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.HasPartitionRequest)
		resp := &milvuspb.BoolResponse{}
		if !ok {
			s, err := BadRequestStatus()
			resp.Status = s
			return resp, err
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.Equal(t, partitionName, req.GetPartitionName())
		resp.Value = false
		s, err := SuccessStatus()
		resp.Status = s
		return resp, err
	})

	assert.Nil(t, c.CreatePartition(ctx, testCollectionName, partitionName))
}

func TestGrpcClientDropPartition(t *testing.T) {
	partitionName := fmt.Sprintf("_part_%d", rand.Int())
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MHasPartition, hasPartitionInjection(t, testCollectionName, true, partitionName)) // injection has assertion of collName & parition name
	assert.Nil(t, c.DropPartition(ctx, testCollectionName, partitionName))
}

func TestGrpcClientHasPartition(t *testing.T) {
	partitionName := fmt.Sprintf("_part_%d", rand.Int())
	ctx := context.Background()
	c := testClient(ctx, t)
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MHasPartition, hasPartitionInjection(t, testCollectionName, false, partitionName)) // injection has assertion of collName & parition name

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
		req, ok := raw.(*milvuspb.ShowPartitionsRequest)
		resp := &milvuspb.ShowPartitionsResponse{}
		if !ok {
			s, err := BadRequestStatus()
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
		s, err := SuccessStatus()
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
		mockServer.SetInjection(MShowPartitions, getPartitionsInterception(t, tc.collName, tc.partitions...))
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

func TestGrpcClientReleasePartitions(t *testing.T) {
	ctx := context.Background()

	c := testClient(ctx, t)

	parts := []string{"_part1", "_part2"}
	mockServer.SetInjection(MHasCollection, hasCollectionDefault)
	mockServer.SetInjection(MHasPartition, hasPartitionInjection(t, testCollectionName, true, "_part1", "_part2", "_part3", "_part4"))
	mockServer.SetInjection(MReleasePartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
		req, ok := raw.(*milvuspb.ReleasePartitionsRequest)
		if !ok {
			return BadRequestStatus()
		}
		assert.Equal(t, testCollectionName, req.GetCollectionName())
		assert.ElementsMatch(t, parts, req.GetPartitionNames())

		return SuccessStatus()
	})
	defer mockServer.SetInjection(MHasPartition, hasPartitionInjection(t, testCollectionName, false, "testPart"))

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
		mockServer.SetInjection(MShowPartitions, getPartitionsInterception(t, testCollectionName, partitions...))
		parts, err := c.ShowPartitions(ctx, testCollectionName)
		assert.NoError(t, err)
		assert.NotNil(t, parts)
	})

	t.Run("bad response", func(t *testing.T) {
		mockServer.SetInjection(MShowPartitions, func(ctx context.Context, raw proto.Message) (proto.Message, error) {
			resp := &milvuspb.ShowPartitionsResponse{}
			resp.PartitionIDs = make([]int64, 0, len(partitions))
			for _, part := range partitions {
				resp.PartitionIDs = append(resp.PartitionIDs, part.ID)
			}
			s, err := SuccessStatus()
			resp.Status = s
			return resp, err
		})
		_, err := c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)
	})

	t.Run("Service error", func(t *testing.T) {
		mockServer.SetInjection(MShowPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &milvuspb.ShowPartitionsResponse{}, errors.New("always fail")
		})
		defer mockServer.DelInjection(MShowPartitions)

		_, err := c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)

		mockServer.SetInjection(MShowPartitions, func(_ context.Context, raw proto.Message) (proto.Message, error) {
			return &milvuspb.ShowPartitionsResponse{
				Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError},
			}, nil
		})
		_, err = c.ShowPartitions(ctx, testCollectionName)
		assert.Error(t, err)
	})
}

type PartitionSuite struct {
	MockSuiteBase
}

func (s *PartitionSuite) TestLoadPartitions() {
	c := s.client
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partNames := []string{"part_1", "part_2"}
	mPartNames := map[string]struct{}{"part_1": {}, "part_2": {}}

	s.Run("normal_run_async", func() {
		defer s.resetMock()
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
			Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.HasPartitionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			_, ok := mPartNames[req.GetPartitionName()]
			s.True(ok)
		}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().LoadPartitions(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadPartitionsRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			s.ElementsMatch(partNames, req.GetPartitionNames())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		err := c.LoadPartitions(ctx, testCollectionName, partNames, true)
		s.NoError(err)
	})

	s.Run("normal_run_sync", func() {
		defer s.resetMock()
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
			Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.HasPartitionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			_, ok := mPartNames[req.GetPartitionName()]
			s.True(ok)
		}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().LoadPartitions(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadPartitionsRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			s.ElementsMatch(partNames, req.GetPartitionNames())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).
			Return(&milvuspb.GetLoadingProgressResponse{Status: &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, Progress: 100}, nil)

		err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
		s.NoError(err)
	})

	s.Run("has_collection_failure", func() {
		s.Run("return_false", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)

			err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
			s.Error(err)
		})

		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(nil, errors.New("mock error"))

			err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
			s.Error(err)
		})
	})

	s.Run("has_partition_failure", func() {
		s.Run("return_false", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: false}, nil)

			err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
			s.Error(err)
		})

		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).
				Return(nil, errors.New("mock"))

			err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
			s.Error(err)
		})
	})

	s.Run("load_partitions_failure", func() {
		s.Run("fail_status_code", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.HasPartitionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				_, ok := mPartNames[req.GetPartitionName()]
				s.True(ok)
			}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().LoadPartitions(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadPartitionsRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				s.ElementsMatch(partNames, req.GetPartitionNames())
			}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

			err := c.LoadPartitions(ctx, testCollectionName, partNames, true)
			s.Error(err)
		})

		s.Run("return_error", func() {
			defer s.resetMock()
			s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
				Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.HasPartitionRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				_, ok := mPartNames[req.GetPartitionName()]
				s.True(ok)
			}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
			s.mock.EXPECT().LoadPartitions(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadPartitionsRequest) {
				s.Equal(testCollectionName, req.GetCollectionName())
				s.ElementsMatch(partNames, req.GetPartitionNames())
			}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)

			err := c.LoadPartitions(ctx, testCollectionName, partNames, true)
			s.Error(err)
		})
	})

	s.Run("get_loading_progress_failure", func() {
		defer s.resetMock()
		s.mock.EXPECT().HasCollection(mock.Anything, &milvuspb.HasCollectionRequest{CollectionName: testCollectionName}).
			Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().HasPartition(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.HasPartitionRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			_, ok := mPartNames[req.GetPartitionName()]
			s.True(ok)
		}).Return(&milvuspb.BoolResponse{Status: &commonpb.Status{}, Value: true}, nil)
		s.mock.EXPECT().LoadPartitions(mock.Anything, mock.Anything).Run(func(_ context.Context, req *milvuspb.LoadPartitionsRequest) {
			s.Equal(testCollectionName, req.GetCollectionName())
			s.ElementsMatch(partNames, req.GetPartitionNames())
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)
		s.mock.EXPECT().GetLoadingProgress(mock.Anything, mock.Anything).
			Return(nil, errors.New("mock error"))

		err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
		s.Error(err)
	})

	s.Run("service_not_ready", func() {
		c := &GrpcClient{}
		err := c.LoadPartitions(ctx, testCollectionName, partNames, false)
		s.ErrorIs(err, ErrClientNotReady)
	})
}

func TestPartitionSuite(t *testing.T) {
	suite.Run(t, new(PartitionSuite))
}
