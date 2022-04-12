package testutil

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

var r *rand.Rand = nil

func init() {
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
}

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

// GenRandomString returns a batch of random string
func GenRandomString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[r.Intn(len(letterRunes))]
	}
	return string(b)
}

func GenDefaultFields(dim int64) []*entity.Field {
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
				entity.TYPE_PARAM_DIM: fmt.Sprintf("%d", dim),
			},
		},
	}
	return fields
}

func GenSchema(name string, autoID bool, fields []*entity.Field) *entity.Schema {
	schema := &entity.Schema{
		CollectionName: name,
		AutoID:         autoID,
		Fields:         fields,
	}
	return schema
}
