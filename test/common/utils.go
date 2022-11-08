package common

import (
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"math/rand"
	"time"

	//"time"
)

const (
	DefaultTimeout = 60

	DefaultIntFieldName = "int64"

	DefaultFloatFieldName = "float"

	DefaultFloatVecFieldName = "float_vec"

	DefaultBinaryVecFieldName = "binary_vec"

	DefaultPartition = "_default"

	DefaultDim = 128

	DefaultShards = int32(2)

	DefaultConsistencyLevel = entity.ConsistencyLevel(0)

	DefaultNb = 3000

	DefaultNq = 5

	DefaultTopK = 10
)

var r *rand.Rand

func init() {
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
}

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

// gen random string
func GenRandomString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[r.Intn(len(letterRunes))]
	}
	return string(b)
}

// gen default fields with int64, float, floatVector field
func GenDefaultFields() []*entity.Field {
	var fields = []*entity.Field{
		{
			Name:       DefaultIntFieldName,
			DataType:   entity.FieldTypeInt64,
			PrimaryKey: true,
		},
		{
			Name:     DefaultFloatFieldName,
			DataType: entity.FieldTypeFloat,
		},
		{
			Name:     DefaultFloatVecFieldName,
			DataType: entity.FieldTypeFloatVector,
			TypeParams: map[string]string{
				entity.TypeParamDim: fmt.Sprintf("%d", DefaultDim),
			},
		},
	}
	return fields
}

// gen schema
func GenSchema(name string, autoID bool, fields []*entity.Field) *entity.Schema {
	schema := &entity.Schema{
		CollectionName: name,
		AutoID:         autoID,
		Fields:         fields,
	}
	return schema
}

// gen float vector values
func GenFloatVector(num, dim int) [][]float32 {
	rand.Seed(time.Now().Unix())
	r := make([][]float32, 0, num)
	for i := 0; i < num; i++ {
		v := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			v = append(v, rand.Float32())
		}
		r = append(r, v)
	}
	return r
}

// gen default column with data
func GenDefaultColumnData( nb int) (*entity.ColumnInt64, *entity.ColumnFloat, *entity.ColumnFloatVector) {
	int64Values := make([]int64, 0, nb)
	floatValues := make([]float32, 0, nb)
	vecFloatValues := make([][]float32, 0, nb)
	for i := 0; i < nb; i++ {
		int64Values = append(int64Values, int64(i))
		floatValues = append(floatValues, float32(i))
		vec := make([]float32, 0, DefaultDim)
		for j := 0; j < DefaultDim; j++ {
			vec = append(vec, rand.Float32())
		}
		vecFloatValues = append(vecFloatValues, vec)
	}
	intColumn := entity.NewColumnInt64(DefaultIntFieldName, int64Values)
	floatColumn := entity.NewColumnFloat(DefaultFloatFieldName, floatValues)
	vecColumn := entity.NewColumnFloatVector(DefaultFloatVecFieldName, DefaultDim, vecFloatValues)
	return intColumn, floatColumn, vecColumn
}

// gen search vectors
func GenSearchVectors(nq int, dim int) []entity.Vector {
	vectors := make([]entity.Vector, 0 , nq)
	for i := 0; i < nq; i++ {
		vector := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vector = append(vector, rand.Float32())
		}
		vectors = append(vectors, entity.FloatVector(vector))
	}
	return vectors
}
