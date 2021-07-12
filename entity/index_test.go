package entity

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGenericIndex(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	name := fmt.Sprintf("generic_index_%d", rand.Int())
	gi := NewGenericIndex(name, IvfFlat, map[string]string{
		tMetricType: string(IP),
	})
	assert.Equal(t, name, gi.Name())
	assert.EqualValues(t, IvfFlat, gi.Params()[tIndexType])
}

func TestFlatIndex(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	name := fmt.Sprintf("flat_index_%d", rand.Int())
	fi := NewFlatIndex(name, L2)
	assert.Equal(t, name, fi.Name())
	assert.EqualValues(t, Flat, fi.Params()[tIndexType])
	assert.EqualValues(t, L2, fi.Params()[tMetricType])
}
