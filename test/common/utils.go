package common

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// const default value for test
const (
	DefaultIntFieldName       = "int64"
	DefaultFloatFieldName     = "float"
	DefaultVarcharFieldName   = "varchar"
	DefaultJSONFieldName      = "json"
	DefaultFloatVecFieldName  = "floatVec"
	DefaultBinaryVecFieldName = "binaryVec"
	DefaultDynamicNumberField = "dynamicNumber"
	DefaultDynamicStringField = "dynamicString"
	DefaultDynamicBoolField   = "dynamicBool"
	DefaultDynamicListField   = "dynamicList"
	RowCount                  = "row_count"
	DefaultTimeout            = 120
	DefaultDim                = int64(128)
	DefaultShards             = int32(2)
	DefaultNb                 = 3000
	DefaultNq                 = 5
	DefaultTopK               = 10
)

// const default value from milvus
const (
	MaxPartitionNum         = 4096
	DefaultDynamicFieldName = "$meta"
	DefaultPartition        = "_default"
	DefaultIndexName        = "_default_idx_102"
	DefaultIndexNameBinary  = "_default_idx_100"
	DefaultRgName           = "__default_resource_group"
	DefaultDb               = "default"
	DefaultConsistencyLevel = entity.ClBounded
	MaxDim                  = 32768
	DefaultMaxLength        = int64(65535)
	MaxCollectionNameLen    = 255
	DefaultRgCapacity       = 1000000
	RetentionDuration       = 40 // common.retentionDuration
)

var IndexStateValue = map[string]int32{
	"IndexStateNone": 0,
	"Unissued":       1,
	"InProgress":     2,
	"Finished":       3,
	"Failed":         4,
	"Retry":          5,
}

var r *rand.Rand

func init() {
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
}

// --- common utils ---
var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

// GenRandomString gen random string
func GenRandomString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[r.Intn(len(letterRunes))]
	}
	return string(b)
}

// GenLongString gen invalid long string
func GenLongString(n int) string {
	var builder strings.Builder
	longString := "a"
	for i := 0; i < n; i++ {
		builder.WriteString(longString)
	}
	return builder.String()
}

// --- common utils  ---

// --- gen fields ---

// GenDefaultFields gen default fields with int64, float, floatVector field
func GenDefaultFields(autoID bool) []*entity.Field {
	intField := GenField(DefaultIntFieldName, entity.FieldTypeInt64, WithIsPrimaryKey(true), WithAutoID(autoID))
	floatField := GenField(DefaultFloatFieldName, entity.FieldTypeFloat)
	floatVecField := GenField(DefaultFloatVecFieldName, entity.FieldTypeFloatVector, WithDim(DefaultDim))
	fields := []*entity.Field{
		intField, floatField, floatVecField,
	}
	return fields
}

// GenDefaultBinaryFields gen default binary fields with int64, float, binaryVector field
func GenDefaultBinaryFields(autoID bool, dim int64) []*entity.Field {
	intField := GenField(DefaultIntFieldName, entity.FieldTypeInt64, WithIsPrimaryKey(true), WithAutoID(autoID))
	floatField := GenField(DefaultFloatFieldName, entity.FieldTypeFloat)
	binaryVecField := GenField(DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, WithDim(dim))

	fields := []*entity.Field{
		intField, floatField, binaryVecField,
	}
	return fields
}

// GenDefaultVarcharFields gen default fields with varchar, floatVector field
func GenDefaultVarcharFields(autoID bool) []*entity.Field {
	varcharField := GenField(DefaultVarcharFieldName, entity.FieldTypeVarChar, WithIsPrimaryKey(true), WithAutoID(autoID), WithMaxLength(DefaultMaxLength))
	binaryVecField := GenField(DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, WithDim(DefaultDim))
	fields := []*entity.Field{
		varcharField, binaryVecField,
	}
	return fields
}

// GenAllFields gen fields with all scala field types
func GenAllFields() []*entity.Field {
	allFields := []*entity.Field{
		GenField("int64", entity.FieldTypeInt64, WithIsPrimaryKey(true)),              // int64
		GenField("bool", entity.FieldTypeBool),                                        // bool
		GenField("int8", entity.FieldTypeInt8),                                        // int8
		GenField("int16", entity.FieldTypeInt16),                                      // int16
		GenField("int32", entity.FieldTypeInt32),                                      // int32
		GenField("float", entity.FieldTypeFloat),                                      // float
		GenField("double", entity.FieldTypeDouble),                                    // double
		GenField("varchar", entity.FieldTypeVarChar, WithMaxLength(DefaultMaxLength)), // varchar
		GenField("json", entity.FieldTypeJSON),                                        // json
		GenField("floatVec", entity.FieldTypeFloatVector, WithDim(DefaultDim)),        // float vector
	}

	return allFields
}

// --- gen fields ---

// --- gen column data ---

// GenDefaultColumnData gen default column with data
func GenDefaultColumnData(start int, nb int, dim int64) (entity.Column, entity.Column, entity.Column) {
	return GenColumnData(start, nb, entity.FieldTypeInt64, DefaultIntFieldName),
		GenColumnData(start, nb, entity.FieldTypeFloat, DefaultFloatFieldName),
		GenColumnData(start, nb, entity.FieldTypeFloatVector, DefaultFloatVecFieldName, WithVectorDim(dim))
}

type GenColumnDataOption func(opt *genDataOpt)

type genDataOpt struct {
	dim int64
}

func WithVectorDim(dim int64) GenColumnDataOption {
	return func(opt *genDataOpt) {
		opt.dim = dim
	}
}

// GenColumnData GenColumnDataOption
func GenColumnData(start int, nb int, fieldType entity.FieldType, fieldName string, opts ...GenColumnDataOption) entity.Column {
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}
	switch fieldType {
	case entity.FieldTypeInt64:
		int64Values := make([]int64, 0, nb)
		for i := start; i < start+nb; i++ {
			int64Values = append(int64Values, int64(i))
		}
		return entity.NewColumnInt64(fieldName, int64Values)

	case entity.FieldTypeInt8:
		int8Values := make([]int8, 0, nb)
		for i := start; i < start+nb; i++ {
			int8Values = append(int8Values, int8(i))
		}
		return entity.NewColumnInt8(fieldName, int8Values)

	case entity.FieldTypeInt16:
		int16Values := make([]int16, 0, nb)
		for i := start; i < start+nb; i++ {
			int16Values = append(int16Values, int16(i))
		}
		return entity.NewColumnInt16(fieldName, int16Values)

	case entity.FieldTypeInt32:
		int32Values := make([]int32, 0, nb)
		for i := start; i < start+nb; i++ {
			int32Values = append(int32Values, int32(i))
		}
		return entity.NewColumnInt32(fieldName, int32Values)

	case entity.FieldTypeBool:
		boolValues := make([]bool, 0, nb)
		for i := start; i < start+nb; i++ {
			boolValues = append(boolValues, i/2 == 0)
		}
		return entity.NewColumnBool(fieldName, boolValues)

	case entity.FieldTypeFloat:
		floatValues := make([]float32, 0, nb)
		for i := start; i < start+nb; i++ {
			floatValues = append(floatValues, float32(i))
		}
		return entity.NewColumnFloat(fieldName, floatValues)

	case entity.FieldTypeDouble:
		floatValues := make([]float64, 0, nb)
		for i := start; i < start+nb; i++ {
			floatValues = append(floatValues, float64(i))
		}
		return entity.NewColumnDouble(fieldName, floatValues)

	case entity.FieldTypeVarChar:
		varcharValues := make([]string, 0, nb)
		for i := start; i < start+nb; i++ {
			varcharValues = append(varcharValues, strconv.Itoa(i))
		}
		return entity.NewColumnVarChar(fieldName, varcharValues)

	case entity.FieldTypeFloatVector:
		vecFloatValues := make([][]float32, 0, nb)
		for i := start; i < start+nb; i++ {
			vec := make([]float32, 0, opt.dim)
			for j := 0; j < int(opt.dim); j++ {
				vec = append(vec, rand.Float32())
			}
			vecFloatValues = append(vecFloatValues, vec)
		}
		return entity.NewColumnFloatVector(fieldName, int(opt.dim), vecFloatValues)
	case entity.FieldTypeBinaryVector:
		binaryVectors := make([][]byte, 0, nb)
		for i := 0; i < nb; i++ {
			vec := make([]byte, opt.dim/8)
			rand.Read(vec)
			binaryVectors = append(binaryVectors, vec)
		}
		return entity.NewColumnBinaryVector(fieldName, int(opt.dim), binaryVectors)
	default:
		return nil
	}
}

// GenDefaultJSONData gen default column with data
func GenDefaultJSONData(columnName string, start int, nb int) *entity.ColumnJSONBytes {
	type JSONStruct struct {
		Number int32   `json:"number" milvus:"name:number"`
		String string  `json:"string" milvus:"name:string"`
		Bool   bool    `json:"bool" milvus:"name:bool"`
		List   []int64 `json:"list" milvus:"name:list"`
	}
	jsonValues := make([][]byte, 0, nb)
	var m JSONStruct
	for i := start; i < start+nb; i++ {
		if i%2 == 0 {
			m = JSONStruct{
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
			}
		} else {
			m = JSONStruct{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}
		}
		bs, err := json.Marshal(&m)
		if err != nil {
			log.Fatalf("Marshal json field failed: %s", err)
		}
		jsonValues = append(jsonValues, bs)
	}
	jsonColumn := entity.NewColumnJSONBytes(columnName, jsonValues)
	return jsonColumn
}

// GenDefaultBinaryData gen default binary collection data
func GenDefaultBinaryData(start int, nb int, dim int64) (entity.Column, entity.Column, entity.Column) {
	return GenColumnData(start, nb, entity.FieldTypeInt64, DefaultIntFieldName),
		GenColumnData(start, nb, entity.FieldTypeFloat, DefaultFloatFieldName),
		GenColumnData(start, nb, entity.FieldTypeBinaryVector, DefaultBinaryVecFieldName, WithVectorDim(dim))
}

func GenDefaultVarcharData(start int, nb int, dim int64) (entity.Column, entity.Column) {
	varColumn := GenColumnData(start, nb, entity.FieldTypeVarChar, DefaultVarcharFieldName)
	binaryColumn := GenColumnData(start, nb, entity.FieldTypeBinaryVector, DefaultBinaryVecFieldName, WithVectorDim(dim))
	return varColumn, binaryColumn
}

func GenAllFieldsData(start int, nb int, dim int64) []entity.Column {
	// prepare data
	data := []entity.Column{
		GenColumnData(start, nb, entity.FieldTypeInt64, "int64"),
		GenColumnData(start, nb, entity.FieldTypeBool, "bool"),
		GenColumnData(start, nb, entity.FieldTypeInt8, "int8"),
		GenColumnData(start, nb, entity.FieldTypeInt16, "int16"),
		GenColumnData(start, nb, entity.FieldTypeInt32, "int32"),
		GenColumnData(start, nb, entity.FieldTypeFloat, "float"),
		GenColumnData(start, nb, entity.FieldTypeDouble, "double"),
		GenColumnData(start, nb, entity.FieldTypeVarChar, "varchar"),
		GenDefaultJSONData("json", start, nb),
		GenColumnData(start, nb, entity.FieldTypeFloatVector, "floatVec", WithVectorDim(dim)),
	}
	return data
}

// --- gen column data ---

// --- gen row data ---

type Dynamic struct {
	Number int32   `json:"dynamicNumber" milvus:"name:dynamicNumber"`
	String string  `json:"dynamicString" milvus:"name:dynamicString"`
	Bool   bool    `json:"dynamicBool" milvus:"name:dynamicBool"`
	List   []int64 `json:"dynamicList" milvus:"name:dynamicList"`
}

func GenDefaultRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	type DynamicRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
		Dynamic  Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
	}

	for i := start; i < start+nb; i++ {
		floatVec := make([]float32, 0, dim)
		for j := 0; j < int(dim); j++ {
			floatVec = append(floatVec, rand.Float32())
		}
		if enableDynamicField {
			var dynamic Dynamic
			if i%2 == 0 {
				dynamic = Dynamic{
					Number: int32(i),
					String: strconv.Itoa(i),
					Bool:   i%2 == 0,
				}
			} else {
				dynamic = Dynamic{
					Number: int32(i),
					String: strconv.Itoa(i),
					Bool:   i%2 == 0,
					List:   []int64{int64(i), int64(i + 1)},
				}
			}

			dynamicRow := DynamicRow{
				Int64:    int64(i),
				Float:    float32(i),
				FloatVec: floatVec,
				Dynamic:  dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &BaseRow{
				Int64:    int64(i),
				Float:    float32(i),
				FloatVec: floatVec,
			})
		}
	}
	return rows
}

func GenDefaultBinaryRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	type DynamicRow struct {
		Int64     int64    `json:"int64" milvus:"name:int64"`
		Float     float32  `json:"float" milvus:"name:float"`
		BinaryVec [][]byte `json:"binaryVec" milvus:"name:binaryVec"`
		Dynamic   Dynamic  `json:"dynamic" milvus:"name:dynamic"`
	}

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64     int64    `json:"int64" milvus:"name:int64"`
		Float     float32  `json:"float" milvus:"name:float"`
		BinaryVec [][]byte `json:"binaryVec" milvus:"name:binaryVec"`
	}

	for i := start; i < start+nb; i++ {
		binaryVec := make([][]byte, 0, nb)
		vec := make([]byte, 0, dim/8)
		rand.Read(vec)
		binaryVec = append(binaryVec, vec)

		if enableDynamicField {
			dynamic := Dynamic{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}

			dynamicRow := DynamicRow{
				Int64:     int64(i),
				Float:     float32(i),
				BinaryVec: binaryVec,
				Dynamic:   dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &BaseRow{
				Int64:     int64(i),
				Float:     float32(i),
				BinaryVec: binaryVec,
			})
		}
	}
	return rows
}

func GenDefaultVarcharRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	type DynamicRow struct {
		Varchar   string   `json:"varchar" milvus:"name:varchar"`
		BinaryVec [][]byte `json:"binaryVec" milvus:"name:binaryVec"`
		Dynamic   Dynamic  `json:"dynamic" milvus:"name:dynamic"`
	}

	// BaseRow generate insert rows
	type BaseRow struct {
		Varchar   string   `json:"varchar" milvus:"name:varchar"`
		BinaryVec [][]byte `json:"binaryVec" milvus:"name:binaryVec"`
	}

	for i := start; i < start+nb; i++ {
		binaryVec := make([][]byte, 0, nb)
		vec := make([]byte, 0, dim/8)
		rand.Read(vec)
		binaryVec = append(binaryVec, vec)

		if enableDynamicField {
			dynamic := Dynamic{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}

			dynamicRow := DynamicRow{
				Varchar:   strconv.Itoa(i),
				BinaryVec: binaryVec,
				Dynamic:   dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &BaseRow{
				Varchar:   strconv.Itoa(i),
				BinaryVec: binaryVec,
			})
		}
	}
	return rows
}

func GenDefaultJSONRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)
	type JSONStruct struct {
		Number int32   `json:"number" milvus:"name:number"`
		String string  `json:"string" milvus:"name:string"`
		Bool   bool    `json:"bool" milvus:"name:bool"`
		List   []int64 `json:"list" milvus:"name:list"`
	}

	type BaseDynamicRow struct {
		Int64    int64      `json:"int64" milvus:"name:int64"`
		Float    float32    `json:"float" milvus:"name:float"`
		FloatVec []float32  `json:"floatVec" milvus:"name:floatVec"`
		JSON     JSONStruct `json:"json" milvus:"name:json"`
		Number   int32      `json:"dynamicNumber" milvus:"name:dynamicNumber"`
		String   string     `json:"dynamicString" milvus:"name:dynamicString"`
		Bool     bool       `json:"dynamicBool" milvus:"name:dynamicBool"`
		//List     []int64    `json:"dynamicList" milvus:"name:dynamicList"`
	}

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64    int64      `json:"int64" milvus:"name:int64"`
		Float    float32    `json:"float" milvus:"name:float"`
		FloatVec []float32  `json:"floatVec" milvus:"name:floatVec"`
		JSON     JSONStruct `json:"json" milvus:"name:json"`
	}

	for i := start; i < start+nb; i++ {
		floatVec := make([]float32, 0, dim)
		for j := 0; j < int(dim); j++ {
			floatVec = append(floatVec, rand.Float32())
		}

		// jsonStruct row and dynamic row
		var jsonStruct JSONStruct
		if i%2 == 0 {
			jsonStruct = JSONStruct{
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
			}
		} else {
			jsonStruct = JSONStruct{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}
		}
		if enableDynamicField {
			baseDynamicRow := BaseDynamicRow{
				Int64:    int64(i),
				Float:    float32(i),
				FloatVec: floatVec,
				JSON:     jsonStruct,
				Number:   int32(i),
				String:   strconv.Itoa(i),
				Bool:     i%2 == 0,
				//List:     []int64{int64(i), int64(i + 1)},
			}

			rows = append(rows, baseDynamicRow)
		} else {
			rows = append(rows, &BaseRow{
				Int64:    int64(i),
				Float:    float32(i),
				FloatVec: floatVec,
				JSON:     jsonStruct,
			})
		}
	}
	return rows
}

func GenAllFieldsRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	type DynamicRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Bool     bool      `json:"bool" milvus:"name:bool"`
		Int8     int8      `json:"int8" milvus:"name:int8"`
		Int16    int16     `json:"int16" milvus:"name:int16"`
		Int32    int32     `json:"int32" milvus:"name:int32"`
		Float    float32   `json:"float" milvus:"name:float"`
		Double   float64   `json:"double" milvus:"name:double"`
		Varchar  string    `json:"varchar" milvus:"name:varchar"`
		JSON     Dynamic   `json:"json" milvus:"name:json"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
		Dynamic  Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Bool     bool      `json:"bool" milvus:"name:bool"`
		Int8     int8      `json:"int8" milvus:"name:int8"`
		Int16    int16     `json:"int16" milvus:"name:int16"`
		Int32    int32     `json:"int32" milvus:"name:int32"`
		Float    float32   `json:"float" milvus:"name:float"`
		Double   float64   `json:"double" milvus:"name:double"`
		Varchar  string    `json:"varchar" milvus:"name:varchar"`
		JSON     Dynamic   `json:"json" milvus:"name:json"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
	}

	for i := start; i < start+nb; i++ {
		floatVec := make([]float32, 0, dim)
		for j := 0; j < int(dim); j++ {
			floatVec = append(floatVec, rand.Float32())
		}

		// json and dynamic field
		dynamicJSON := Dynamic{
			Number: int32(i),
			String: strconv.Itoa(i),
			Bool:   i%2 == 0,
			List:   []int64{int64(i), int64(i + 1)},
		}
		if enableDynamicField {
			dynamicRow := DynamicRow{
				Int64:    int64(i),
				Bool:     i%2 == 0,
				Int8:     int8(i),
				Int16:    int16(i),
				Int32:    int32(i),
				Float:    float32(i),
				Double:   float64(i),
				Varchar:  strconv.Itoa(i),
				FloatVec: floatVec,
				JSON:     dynamicJSON,
				Dynamic:  dynamicJSON,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &BaseRow{
				Int64:    int64(i),
				Bool:     i%2 == 0,
				Int8:     int8(i),
				Int16:    int16(i),
				Int32:    int32(i),
				Float:    float32(i),
				Double:   float64(i),
				Varchar:  strconv.Itoa(i),
				FloatVec: floatVec,
				JSON:     dynamicJSON,
			})
		}
	}
	return rows
}

func GenDynamicFieldData(start int, nb int) []entity.Column {
	type ListStruct struct {
		List []int64 `json:"list" milvus:"name:list"`
	}

	// gen number, string bool list data column
	numberValues := make([]int32, 0, nb)
	stringValues := make([]string, 0, nb)
	boolValues := make([]bool, 0, nb)
	//listValues := make([][]byte, 0, nb)
	//m := make(map[string]interface{})
	for i := start; i < start+nb; i++ {
		numberValues = append(numberValues, int32(i))
		stringValues = append(stringValues, strconv.Itoa(i))
		boolValues = append(boolValues, i%3 == 0)
		//m["list"] = ListStruct{
		//	List: []int64{int64(i), int64(i + 1)},
		//}
		//bs, err := json.Marshal(m)
		//if err != nil {
		//	log.Fatalf("Marshal json field failed: %s", err)
		//}
		//listValues = append(listValues, bs)
	}
	data := []entity.Column{
		entity.NewColumnInt32(DefaultDynamicNumberField, numberValues),
		entity.NewColumnString(DefaultDynamicStringField, stringValues),
		entity.NewColumnBool(DefaultDynamicBoolField, boolValues),
		//entity.NewColumnJSONBytes(DefaultDynamicListField, listValues),
	}
	return data
}

func MergeColumnsToDynamic(nb int, columns []entity.Column) *entity.ColumnJSONBytes {
	values := make([][]byte, 0, nb)
	for i := 0; i < nb; i++ {
		m := make(map[string]interface{})
		for _, column := range columns {
			// range guaranteed
			m[column.Name()], _ = column.Get(i)
		}
		bs, err := json.Marshal(&m)
		if err != nil {
			log.Fatal(err)
		}
		values = append(values, bs)
	}
	jsonColumn := entity.NewColumnJSONBytes(DefaultDynamicFieldName, values)

	var jsonData []string
	for i := 0; i < jsonColumn.Len(); i++ {
		line, err := jsonColumn.GetAsString(i)
		if err != nil {
			fmt.Println(err)
		}
		jsonData = append(jsonData, line)
	}

	return jsonColumn
}

// --- gen row data ---

// --- index utils ---

// GenAllFloatIndex gen all float vector index
func GenAllFloatIndex(metricType entity.MetricType) []entity.Index {
	nlist := 128
	idxFlat, _ := entity.NewIndexFlat(metricType)
	idxIvfFlat, _ := entity.NewIndexIvfFlat(metricType, nlist)
	idxIvfSq8, _ := entity.NewIndexIvfSQ8(metricType, nlist)
	idxIvfPq, _ := entity.NewIndexIvfPQ(metricType, nlist, 16, 8)
	idxHnsw, _ := entity.NewIndexHNSW(metricType, 8, 96)
	idxDiskAnn, _ := entity.NewIndexDISKANN(metricType)

	allFloatIndex := []entity.Index{
		idxFlat,
		idxIvfFlat,
		idxIvfSq8,
		idxIvfPq,
		idxHnsw,
		idxDiskAnn,
	}
	return allFloatIndex
}

// --- index utils ---

// --- search utils ---

// GenSearchVectors gen search vectors
func GenSearchVectors(nq int, dim int64, dataType entity.FieldType) []entity.Vector {
	vectors := make([]entity.Vector, 0, nq)
	switch dataType {
	case entity.FieldTypeFloatVector:
		for i := 0; i < nq; i++ {
			vector := make([]float32, 0, dim)
			for j := 0; j < int(dim); j++ {
				vector = append(vector, rand.Float32())
			}
			vectors = append(vectors, entity.FloatVector(vector))
		}
	case entity.FieldTypeBinaryVector:
		for i := 0; i < nq; i++ {
			vector := make([]byte, dim/8)
			rand.Read(vector)
			vectors = append(vectors, entity.BinaryVector(vector))
		}
	}
	return vectors
}

// --- search utils ---
