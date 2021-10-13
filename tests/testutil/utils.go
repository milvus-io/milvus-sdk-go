package testutil

import (
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"math/rand"
	"time"
)

var r *rand.Rand = nil

func init() {
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
}

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

// RandomString returns a batch of random string
func GenRandomString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[r.Intn(len(letterRunes))]
	}
	return string(b)
}

func GenDefaultFields(dim int64) []*entity.Field {
	var fields = []*entity.Field {
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
			TypeParams: map[string]string {
				"dim": fmt.Sprintf("%d", dim),
			},
		},
	}
	return fields
}

func GenSchema(name string, autoId bool, fields []*entity.Field) *entity.Schema {
	schema := &entity.Schema{
		CollectionName: name,
		AutoID: autoId,
		Fields: fields,
	}
	return schema
}