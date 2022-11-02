// Code generated by go generate; DO NOT EDIT
// This file is generated by go generate at 2022-11-02 17:52:36.293259532 +0800 CST m=+0.004066234

package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)


func TestIndexFlat(t *testing.T){
	

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		
		idx0, err := NewIndexFlat(mt, 
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "Flat", idx0.Name())
		assert.EqualValues(t, "FLAT", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
	})
}

func TestIndexBinFlat(t *testing.T){
	
	var nlist int

	mt := HAMMING
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist = 10
		idx0, err := NewIndexBinFlat(mt, 
			nlist,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "BinFlat", idx0.Name())
		assert.EqualValues(t, "BIN_FLAT", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.True(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist = 0
		idx0, err := NewIndexBinFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist = 65537
		idx1, err := NewIndexBinFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexIvfFlat(t *testing.T){
	
	var nlist int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist = 10
		idx0, err := NewIndexIvfFlat(mt, 
			nlist,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "IvfFlat", idx0.Name())
		assert.EqualValues(t, "IVF_FLAT", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist = 0
		idx0, err := NewIndexIvfFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist = 65537
		idx1, err := NewIndexIvfFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexBinIvfFlat(t *testing.T){
	
	var nlist int

	mt := HAMMING
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist = 10
		idx0, err := NewIndexBinIvfFlat(mt, 
			nlist,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "BinIvfFlat", idx0.Name())
		assert.EqualValues(t, "BIN_IVF_FLAT", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.True(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist = 0
		idx0, err := NewIndexBinIvfFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist = 65537
		idx1, err := NewIndexBinIvfFlat(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexIvfSQ8(t *testing.T){
	
	var nlist int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist = 10
		idx0, err := NewIndexIvfSQ8(mt, 
			nlist,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "IvfSQ8", idx0.Name())
		assert.EqualValues(t, "IVF_SQ8", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist = 0
		idx0, err := NewIndexIvfSQ8(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist = 65537
		idx1, err := NewIndexIvfSQ8(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexIvfSQ8H(t *testing.T){
	
	var nlist int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist = 10
		idx0, err := NewIndexIvfSQ8H(mt, 
			nlist,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "IvfSQ8H", idx0.Name())
		assert.EqualValues(t, "IVF_SQ8_HYBRID", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist = 0
		idx0, err := NewIndexIvfSQ8H(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist = 65537
		idx1, err := NewIndexIvfSQ8H(mt, 
			nlist,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexIvfPQ(t *testing.T){
	
	var nlist int
	var m int
	var nbits int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist, m, nbits = 10, 8, 8
		idx0, err := NewIndexIvfPQ(mt, 
			nlist,
			m,
			nbits,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "IvfPQ", idx0.Name())
		assert.EqualValues(t, "IVF_PQ", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist, m, nbits = 0, 8, 8
		idx0, err := NewIndexIvfPQ(mt, 
			nlist,
			m,
			nbits,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist, m, nbits = 65537, 8, 8
		idx1, err := NewIndexIvfPQ(mt, 
			nlist,
			m,
			nbits,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		nlist, m, nbits = 10, 8, 0
		idx2, err := NewIndexIvfPQ(mt, 
			nlist,
			m,
			nbits,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		nlist, m, nbits = 10, 8, 17
		idx3, err := NewIndexIvfPQ(mt, 
			nlist,
			m,
			nbits,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
}

func TestIndexRNSG(t *testing.T){
	
	var out_degree int
	var candidate_pool_size int
	var search_length int
	var knng int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		out_degree, candidate_pool_size, search_length, knng = 30, 300, 60, 50
		idx0, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "RNSG", idx0.Name())
		assert.EqualValues(t, "NSG", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		out_degree, candidate_pool_size, search_length, knng = 4, 300, 60, 50
		idx0, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		out_degree, candidate_pool_size, search_length, knng = 301, 300, 60, 50
		idx1, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 49, 60, 50
		idx2, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 1001, 60, 50
		idx3, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 300, 9, 50
		idx4, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx4)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 300, 301, 50
		idx5, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx5)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 300, 60, 4
		idx6, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx6)
		
		out_degree, candidate_pool_size, search_length, knng = 30, 300, 60, 301
		idx7, err := NewIndexRNSG(mt, 
			out_degree,
			candidate_pool_size,
			search_length,
			knng,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx7)
		
	})
}

func TestIndexHNSW(t *testing.T){
	
	var M int
	var efConstruction int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		M, efConstruction = 16, 40
		idx0, err := NewIndexHNSW(mt, 
			M,
			efConstruction,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "HNSW", idx0.Name())
		assert.EqualValues(t, "HNSW", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		M, efConstruction = 3, 40
		idx0, err := NewIndexHNSW(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		M, efConstruction = 65, 40
		idx1, err := NewIndexHNSW(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		M, efConstruction = 16, 7
		idx2, err := NewIndexHNSW(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		M, efConstruction = 16, 513
		idx3, err := NewIndexHNSW(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
}

func TestIndexRHNSWFlat(t *testing.T){
	
	var M int
	var efConstruction int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		M, efConstruction = 16, 40
		idx0, err := NewIndexRHNSWFlat(mt, 
			M,
			efConstruction,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "RHNSWFlat", idx0.Name())
		assert.EqualValues(t, "RHNSW_FLAT", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		M, efConstruction = 3, 40
		idx0, err := NewIndexRHNSWFlat(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		M, efConstruction = 65, 40
		idx1, err := NewIndexRHNSWFlat(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		M, efConstruction = 16, 7
		idx2, err := NewIndexRHNSWFlat(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		M, efConstruction = 16, 513
		idx3, err := NewIndexRHNSWFlat(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
}

func TestIndexRHNSW_PQ(t *testing.T){
	
	var M int
	var efConstruction int
	var PQM int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		M, efConstruction, PQM = 17, 40, 8
		idx0, err := NewIndexRHNSW_PQ(mt, 
			M,
			efConstruction,
			PQM,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "RHNSW_PQ", idx0.Name())
		assert.EqualValues(t, "RHNSW_PQ", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		M, efConstruction, PQM = 3, 40, 8
		idx0, err := NewIndexRHNSW_PQ(mt, 
			M,
			efConstruction,
			PQM,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		M, efConstruction, PQM = 65, 40, 8
		idx1, err := NewIndexRHNSW_PQ(mt, 
			M,
			efConstruction,
			PQM,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		M, efConstruction, PQM = 16, 7, 8
		idx2, err := NewIndexRHNSW_PQ(mt, 
			M,
			efConstruction,
			PQM,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		M, efConstruction, PQM = 16, 513, 8
		idx3, err := NewIndexRHNSW_PQ(mt, 
			M,
			efConstruction,
			PQM,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
}

func TestIndexRHNSW_SQ(t *testing.T){
	
	var M int
	var efConstruction int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		M, efConstruction = 16, 40
		idx0, err := NewIndexRHNSW_SQ(mt, 
			M,
			efConstruction,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "RHNSW_SQ", idx0.Name())
		assert.EqualValues(t, "RHNSW_SQ", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		M, efConstruction = 3, 40
		idx0, err := NewIndexRHNSW_SQ(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		M, efConstruction = 65, 40
		idx1, err := NewIndexRHNSW_SQ(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		M, efConstruction = 16, 7
		idx2, err := NewIndexRHNSW_SQ(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		M, efConstruction = 16, 513
		idx3, err := NewIndexRHNSW_SQ(mt, 
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
}

func TestIndexIvfHNSW(t *testing.T){
	
	var nlist int
	var M int
	var efConstruction int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		nlist, M, efConstruction = 10, 16, 40
		idx0, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "IvfHNSW", idx0.Name())
		assert.EqualValues(t, "IVF_HNSW", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		nlist, M, efConstruction = 0, 16, 40
		idx0, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nlist, M, efConstruction = 65537, 16, 40
		idx1, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		nlist, M, efConstruction = 10, 3, 40
		idx2, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		nlist, M, efConstruction = 10, 65, 40
		idx3, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
		nlist, M, efConstruction = 10, 16, 7
		idx4, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx4)
		
		nlist, M, efConstruction = 10, 16, 513
		idx5, err := NewIndexIvfHNSW(mt, 
			nlist,
			M,
			efConstruction,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx5)
		
	})
}

func TestIndexANNOY(t *testing.T){
	
	var n_trees int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		n_trees = 8
		idx0, err := NewIndexANNOY(mt, 
			n_trees,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "ANNOY", idx0.Name())
		assert.EqualValues(t, "ANNOY", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		n_trees = 0
		idx0, err := NewIndexANNOY(mt, 
			n_trees,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		n_trees = 1025
		idx1, err := NewIndexANNOY(mt, 
			n_trees,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
}

func TestIndexNGTPANNG(t *testing.T){
	
	var edge_size int
	var forcedly_pruned_edge_size int
	var selectively_pruned_edge_size int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 20, 40, 10
		idx0, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "NGTPANNG", idx0.Name())
		assert.EqualValues(t, "NGT_PANNG", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 0, 40, 10
		idx0, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 201, 40, 10
		idx1, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 20, 10, 10
		idx2, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 20, 201, 10
		idx3, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 20, 40, 40
		idx4, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx4)
		
		edge_size, forcedly_pruned_edge_size, selectively_pruned_edge_size = 20, 40, 201
		idx5, err := NewIndexNGTPANNG(mt, 
			edge_size,
			forcedly_pruned_edge_size,
			selectively_pruned_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx5)
		
	})
}

func TestIndexNGTONNG(t *testing.T){
	
	var edge_size int
	var outgoing_edge_size int
	var incoming_edge_size int

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		edge_size, outgoing_edge_size, incoming_edge_size = 20, 10, 40
		idx0, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "NGTONNG", idx0.Name())
		assert.EqualValues(t, "NGT_ONNG", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
		edge_size, outgoing_edge_size, incoming_edge_size = 0, 10, 40
		idx0, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		edge_size, outgoing_edge_size, incoming_edge_size = 201, 10, 40
		idx1, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		edge_size, outgoing_edge_size, incoming_edge_size = 20, 0, 40
		idx2, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		edge_size, outgoing_edge_size, incoming_edge_size = 20, 201, 40
		idx3, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
		edge_size, outgoing_edge_size, incoming_edge_size = 20, 10, 0
		idx4, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx4)
		
		edge_size, outgoing_edge_size, incoming_edge_size = 20, 10, 201
		idx5, err := NewIndexNGTONNG(mt, 
			edge_size,
			outgoing_edge_size,
			incoming_edge_size,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx5)
		
	})
}

func TestIndexAUTOINDEX(t *testing.T){
	

	mt := L2
	

	t.Run("valid usage case", func(t *testing.T){
		
		
		idx0, err := NewIndexAUTOINDEX(mt, 
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.Equal(t, "AUTOINDEX", idx0.Name())
		assert.EqualValues(t, "AUTOINDEX", idx0.IndexType())
		assert.NotNil(t, idx0.Params())
		assert.False(t, idx0.SupportBinary())
		
	})

	t.Run("invalid usage case", func(t *testing.T){
		
	})
}

