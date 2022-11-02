// Code generated by go generate; DO NOT EDIT
// This file is generated by go generate at 2022-11-02 17:52:36.293259532 +0800 CST m=+0.004066234

package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)


func TestIndexFlatSearchParam(t *testing.T) {
	

	t.Run("valid usage case", func(t *testing.T){
		
		
		idx0, err := NewIndexFlatSearchParam(
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
}

func TestIndexBinFlatSearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexBinFlatSearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexBinFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexBinFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexIvfFlatSearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexIvfFlatSearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexIvfFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexIvfFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexBinIvfFlatSearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexBinIvfFlatSearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexBinIvfFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexBinIvfFlatSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexIvfSQ8SearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexIvfSQ8SearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexIvfSQ8SearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexIvfSQ8SearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexIvfSQ8HSearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexIvfSQ8HSearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexIvfSQ8HSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexIvfSQ8HSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexIvfPQSearchParam(t *testing.T) {
	
	var nprobe int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe = 10
		idx0, err := NewIndexIvfPQSearchParam(
			nprobe,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe = 0
		idx0, err := NewIndexIvfPQSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe = 65537
		idx1, err := NewIndexIvfPQSearchParam(
			nprobe,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexRNSGSearchParam(t *testing.T) {
	
	var search_length int

	t.Run("valid usage case", func(t *testing.T){
		
		search_length = 100
		idx0, err := NewIndexRNSGSearchParam(
			search_length,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		search_length = 9
		idx0, err := NewIndexRNSGSearchParam(
			search_length,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		search_length = 301
		idx1, err := NewIndexRNSGSearchParam(
			search_length,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexHNSWSearchParam(t *testing.T) {
	
	var ef int

	t.Run("valid usage case", func(t *testing.T){
		
		ef = 16
		idx0, err := NewIndexHNSWSearchParam(
			ef,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		ef = 0
		idx0, err := NewIndexHNSWSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		ef = 32769
		idx1, err := NewIndexHNSWSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexRHNSWFlatSearchParam(t *testing.T) {
	
	var ef int

	t.Run("valid usage case", func(t *testing.T){
		
		ef = 16
		idx0, err := NewIndexRHNSWFlatSearchParam(
			ef,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		ef = 0
		idx0, err := NewIndexRHNSWFlatSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		ef = 32769
		idx1, err := NewIndexRHNSWFlatSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexRHNSW_PQSearchParam(t *testing.T) {
	
	var ef int

	t.Run("valid usage case", func(t *testing.T){
		
		ef = 16
		idx0, err := NewIndexRHNSW_PQSearchParam(
			ef,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		ef = 0
		idx0, err := NewIndexRHNSW_PQSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		ef = 32769
		idx1, err := NewIndexRHNSW_PQSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexRHNSW_SQSearchParam(t *testing.T) {
	
	var ef int

	t.Run("valid usage case", func(t *testing.T){
		
		ef = 16
		idx0, err := NewIndexRHNSW_SQSearchParam(
			ef,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		ef = 0
		idx0, err := NewIndexRHNSW_SQSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		ef = 32769
		idx1, err := NewIndexRHNSW_SQSearchParam(
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
	})
	
}

func TestIndexIvfHNSWSearchParam(t *testing.T) {
	
	var nprobe int
	var ef int

	t.Run("valid usage case", func(t *testing.T){
		
		nprobe, ef = 10, 16
		idx0, err := NewIndexIvfHNSWSearchParam(
			nprobe,
			ef,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		nprobe, ef = 0, 16
		idx0, err := NewIndexIvfHNSWSearchParam(
			nprobe,
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		nprobe, ef = 65537, 16
		idx1, err := NewIndexIvfHNSWSearchParam(
			nprobe,
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		nprobe, ef = 10, 0
		idx2, err := NewIndexIvfHNSWSearchParam(
			nprobe,
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		nprobe, ef = 10, 32769
		idx3, err := NewIndexIvfHNSWSearchParam(
			nprobe,
			ef,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
	
}

func TestIndexANNOYSearchParam(t *testing.T) {
	
	var search_k int

	t.Run("valid usage case", func(t *testing.T){
		
		search_k = -1
		idx0, err := NewIndexANNOYSearchParam(
			search_k,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
		search_k = 20
		idx1, err := NewIndexANNOYSearchParam(
			search_k,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx1)
		assert.NotNil(t, idx1.Params())
		
	})
	
}

func TestIndexNGTPANNGSearchParam(t *testing.T) {
	
	var max_search_edges int
	var epsilon float64

	t.Run("valid usage case", func(t *testing.T){
		
		max_search_edges, epsilon = 40, 0.1
		idx0, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
		max_search_edges, epsilon = -1, -0.5
		idx1, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx1)
		assert.NotNil(t, idx1.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		max_search_edges, epsilon = -2, 0.1
		idx0, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		max_search_edges, epsilon = 201, 0.1
		idx1, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		max_search_edges, epsilon = 40, 1.2
		idx2, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		max_search_edges, epsilon = 40, -1.1
		idx3, err := NewIndexNGTPANNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
	
}

func TestIndexNGTONNGSearchParam(t *testing.T) {
	
	var max_search_edges int
	var epsilon float64

	t.Run("valid usage case", func(t *testing.T){
		
		max_search_edges, epsilon = 40, 0.1
		idx0, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
		max_search_edges, epsilon = -1, -0.5
		idx1, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx1)
		assert.NotNil(t, idx1.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		max_search_edges, epsilon = -2, 0.1
		idx0, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		max_search_edges, epsilon = 201, 0.1
		idx1, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		max_search_edges, epsilon = 40, 1.2
		idx2, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
		max_search_edges, epsilon = 40, -1.1
		idx3, err := NewIndexNGTONNGSearchParam(
			max_search_edges,
			epsilon,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx3)
		
	})
	
}

func TestIndexAUTOINDEXSearchParam(t *testing.T) {
	
	var level int

	t.Run("valid usage case", func(t *testing.T){
		
		level = 1
		idx0, err := NewIndexAUTOINDEXSearchParam(
			level,
		)
		assert.Nil(t, err)
		assert.NotNil(t, idx0)
		assert.NotNil(t, idx0.Params())
		
	})
	
	t.Run("invalid usage case", func(t *testing.T){
		
		level = 0
		idx0, err := NewIndexAUTOINDEXSearchParam(
			level,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx0)
		
		level = 10
		idx1, err := NewIndexAUTOINDEXSearchParam(
			level,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx1)
		
		level = -1
		idx2, err := NewIndexAUTOINDEXSearchParam(
			level,
		)
		assert.NotNil(t, err)
		assert.Nil(t, idx2)
		
	})
	
}

