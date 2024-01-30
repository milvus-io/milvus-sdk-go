package entity

import (
	"encoding/json"
	"strconv"

	"github.com/cockroachdb/errors"
)

var _ Index = &IndexGPUCagra{}

// IndexGPUCagra index type for GPU Cagra index.
type IndexGPUCagra struct {
	metricType              MetricType
	intermediateGraphDegree int
	graphDegree             int
}

// NewIndexGPUCagra returns an Index with GPU_CAGRA type.
//   - intermediate_graph_degree:  The number of k-nearest neighbors (k) of this intermediate k-NN graph, trade off the quality of the final searchable CAGRA graph;
//   - graph_degree: CAGRA's optimized graph fixed-degree number;
func NewIndexGPUCagra(metricType MetricType,
	intermediateGraphDegree, graphDegree int) (*IndexGPUCagra, error) {

	if intermediateGraphDegree < graphDegree {
		return nil, errors.New("Graph degree cannot be larger than intermediate graph degree")
	}

	return &IndexGPUCagra{
		metricType:              metricType,
		intermediateGraphDegree: intermediateGraphDegree,
		graphDegree:             graphDegree,
	}, nil
}

// Name returns index type name, implementing Index interface
func (i *IndexGPUCagra) Name() string {
	return "GPUCagra"
}

// IndexType returns IndexType, implementing Index interface
func (i *IndexGPUCagra) IndexType() IndexType {
	return GPUCagra
}

// SupportBinary returns whether index type support binary vector
func (i *IndexGPUCagra) SupportBinary() bool {
	return false
}

// Params returns index construction params, implementing Index interface
func (i *IndexGPUCagra) Params() map[string]string {
	params := map[string]string{ //auto generated mapping
		"intermediate_graph_degree": strconv.FormatInt(int64(i.intermediateGraphDegree), 10),
		"graph_degree":              strconv.FormatInt(int64(i.graphDegree), 10),
	}
	bs, _ := json.Marshal(params)
	return map[string]string{
		"params":      string(bs),
		"index_type":  string(i.IndexType()),
		"metric_type": string(i.metricType),
	}
}

type IndexGPUCagraSearchParam struct {
	baseSearchParams
}

// - itopk_size: the main parameter that can be adjusted to trade off search speed, which specifies the size of an internal sorted list that stores the nodes that can be explored in the next iteration;
// - search_width: the number of the closest parent vertices that are traversed to expand their children in each search iteration;
// - min_iterations: Lower limit of search iterations;
// - max_iterations: Upper limit of search iterations. Auto select when 0;
// - team_size: Number of threads used to calculate a single distance.
func NewIndexGPUCagraSearchParam(
	itopkSize int,
	searchWidth int,
	minIterations int,
	maxIterations int,
	teamSize int,
) (*IndexGPUCagraSearchParam, error) {

	if !(teamSize == 0 || teamSize == 4 || teamSize == 8 || teamSize == 16 || teamSize == 32) {
		return nil, errors.New("teamSize shall be 0, 4, 8, 16 or 32")
	}

	sp := &IndexGPUCagraSearchParam{
		baseSearchParams: newBaseSearchParams(),
	}

	sp.params["itopk_size"] = itopkSize
	sp.params["search_width"] = searchWidth
	sp.params["min_iterations"] = minIterations
	sp.params["max_iterations"] = maxIterations
	sp.params["team_size"] = teamSize

	return sp, nil
}

// IndexGPUBruteForce index type for GPU brute force search.
type IndexGPUBruteForce struct {
	metricType MetricType
}

func NewIndexGPUBruteForce(metricType MetricType) (*IndexGPUBruteForce, error) {
	return &IndexGPUBruteForce{
		metricType: metricType,
	}, nil
}

// Name returns index type name, implementing Index interface
func (i *IndexGPUBruteForce) Name() string {
	return "GPUBruteForce"
}

// IndexType returns IndexType, implementing Index interface
func (i *IndexGPUBruteForce) IndexType() IndexType {
	return GPUBruteForce
}

// SupportBinary returns whether index type support binary vector
func (i *IndexGPUBruteForce) SupportBinary() bool {
	return false
}

// Params returns index construction params, implementing Index interface
func (i *IndexGPUBruteForce) Params() map[string]string {
	return map[string]string{
		"params":      "{}",
		"index_type":  string(i.IndexType()),
		"metric_type": string(i.metricType),
	}
}

type IndexGPUBruteForceSearchParam struct {
	baseSearchParams
}

func NewIndexGPUBruteForceSearchParam() (*IndexGPUBruteForceSearchParam, error) {
	return &IndexGPUBruteForceSearchParam{
		baseSearchParams: newBaseSearchParams(),
	}, nil
}
