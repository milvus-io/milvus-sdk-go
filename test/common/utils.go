package common

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// const default value for test
const (
	DefaultIntFieldName         = "int64"
	DefaultInt8FieldName        = "int8"
	DefaultInt16FieldName       = "int16"
	DefaultInt32FieldName       = "int32"
	DefaultBoolFieldName        = "bool"
	DefaultFloatFieldName       = "float"
	DefaultDoubleFieldName      = "double"
	DefaultVarcharFieldName     = "varchar"
	DefaultJSONFieldName        = "json"
	DefaultArrayFieldName       = "array"
	DefaultFloatVecFieldName    = "floatVec"
	DefaultBinaryVecFieldName   = "binaryVec"
	DefaultFloat16VecFieldName  = "fp16Vec"
	DefaultBFloat16VecFieldName = "bf16Vec"
	DefaultDynamicNumberField   = "dynamicNumber"
	DefaultDynamicStringField   = "dynamicString"
	DefaultDynamicBoolField     = "dynamicBool"
	DefaultDynamicListField     = "dynamicList"
	DefaultBoolArrayField       = "boolArray"
	DefaultInt8ArrayField       = "int8Array"
	DefaultInt16ArrayField      = "int16Array"
	DefaultInt32ArrayField      = "int32Array"
	DefaultInt64ArrayField      = "int64Array"
	DefaultFloatArrayField      = "floatArray"
	DefaultDoubleArrayField     = "doubleArray"
	DefaultVarcharArrayField    = "varcharArray"
	RowCount                    = "row_count"
	DefaultTimeout              = 120
	DefaultDim                  = int64(128)
	DefaultShards               = int32(2)
	DefaultNb                   = 3000
	DefaultNq                   = 5
	DefaultTopK                 = 10
	TestCapacity                = 100 // default array field capacity
	TestMaxLen                  = 100 // default varchar field max length
)

// const default value from milvus
const (
	MaxPartitionNum         = 4096
	DefaultDynamicFieldName = "$meta"
	QueryCountFieldName     = "count(*)"
	DefaultPartition        = "_default"
	DefaultIndexName        = "_default_idx_102"
	DefaultIndexNameBinary  = "_default_idx_100"
	DefaultRgName           = "__default_resource_group"
	DefaultDb               = "default"
	DefaultConsistencyLevel = entity.ClBounded
	MaxDim                  = 32768
	MaxLength               = int64(65535)
	MaxCollectionNameLen    = 255
	DefaultRgCapacity       = 1000000
	RetentionDuration       = 40   // common.retentionDuration
	MaxCapacity             = 4096 // max array capacity
	DefaultPartitionNum     = 64   // default num_partitions
	MaxTopK                 = 16384
	MaxVectorFieldNum       = 4
)

var IndexStateValue = map[string]int32{
	"IndexStateNone": 0,
	"Unissued":       1,
	"InProgress":     2,
	"Finished":       3,
	"Failed":         4,
	"Retry":          5,
}

var ArrayFieldType = []entity.FieldType{
	entity.FieldTypeBool,
	entity.FieldTypeInt8,
	entity.FieldTypeInt16,
	entity.FieldTypeInt32,
	entity.FieldTypeInt64,
	entity.FieldTypeFloat,
	entity.FieldTypeDouble,
	//entity.FieldTypeVarChar, //t.Skip("Waiting for varchar bytes array fixed")
}

var AllArrayFieldsName = []string{
	DefaultBoolArrayField,
	DefaultInt8ArrayField,
	DefaultInt16ArrayField,
	DefaultInt32ArrayField,
	DefaultInt64ArrayField,
	DefaultFloatArrayField,
	DefaultDoubleArrayField,
	DefaultVarcharArrayField,
}

var AllVectorsFieldsName = []string{
	DefaultFloatVecFieldName,
	DefaultBinaryVecFieldName,
	DefaultFloat16VecFieldName,
	DefaultBFloat16VecFieldName,
}

// return all fields name
func GetAllFieldsName(enableDynamicField bool) []string {
	allFieldName := []string{
		DefaultIntFieldName,
		DefaultBoolFieldName,
		DefaultInt8FieldName,
		DefaultInt16FieldName,
		DefaultInt32FieldName,
		DefaultFloatFieldName,
		DefaultDoubleFieldName,
		DefaultVarcharFieldName,
		DefaultJSONFieldName,
	}
	allFieldName = append(allFieldName, AllArrayFieldsName...)
	allFieldName = append(allFieldName, AllVectorsFieldsName...)
	if enableDynamicField {
		allFieldName = append(allFieldName, DefaultDynamicFieldName)
	}
	return allFieldName
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

// ColumnIndexFunc generate column index
func ColumnIndexFunc(data []entity.Column, fieldName string) int {
	for index, column := range data {
		if column.Name() == fieldName {
			return index
		}
	}
	return -1
}

func GenFloatVector(dim int64) []float32 {
	vector := make([]float32, 0, dim)
	for j := 0; j < int(dim); j++ {
		vector = append(vector, rand.Float32())
	}
	return vector
}

func GenFloat16Vector(dim int64) []byte {
	vector := make([]byte, 0, dim)
	fp16Data := make([]byte, 2)
	for i := 0; i < int(dim); i++ {
		f := rand.Float32()
		u32 := *(*uint32)(unsafe.Pointer(&f))
		binary.LittleEndian.PutUint16(fp16Data, uint16(u32>>16))
		vector = append(vector, fp16Data...)
	}
	return vector
}

func GenBFloat16Vector(dim int64) []byte {
	vector := make([]byte, 0, dim)
	bf16Data := make([]byte, 2)
	for i := 0; i < int(dim); i++ {
		f := rand.Float32()
		u32 := *(*uint32)(unsafe.Pointer(&f))
		binary.LittleEndian.PutUint16(bf16Data, uint16(u32>>16))
		vector = append(vector, bf16Data...)
	}
	return vector
}

func GenBinaryVector(dim int64) []byte {
	vector := make([]byte, dim/8)
	rand.Read(vector)
	return vector
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
	varcharField := GenField(DefaultVarcharFieldName, entity.FieldTypeVarChar, WithIsPrimaryKey(true), WithAutoID(autoID), WithMaxLength(MaxLength))
	binaryVecField := GenField(DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, WithDim(DefaultDim))
	fields := []*entity.Field{
		varcharField, binaryVecField,
	}
	return fields
}

func GenAllArrayFields() []*entity.Field {
	return GenAllArrayFieldsWithCapacity(TestCapacity)
}

// GenAllArrayFieldsWithCapacity GenAllArrayFields gen all array fields
func GenAllArrayFieldsWithCapacity(capacity int64) []*entity.Field {
	fields := []*entity.Field{
		GenField(DefaultBoolArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeBool), WithMaxCapacity(capacity)),
		GenField(DefaultInt8ArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeInt8), WithMaxCapacity(capacity)),
		GenField(DefaultInt16ArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeInt16), WithMaxCapacity(capacity)),
		GenField(DefaultInt32ArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeInt32), WithMaxCapacity(capacity)),
		GenField(DefaultInt64ArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeInt64), WithMaxCapacity(capacity)),
		GenField(DefaultFloatArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeFloat), WithMaxCapacity(capacity)),
		GenField(DefaultDoubleArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeDouble), WithMaxCapacity(capacity)),
		GenField(DefaultVarcharArrayField, entity.FieldTypeArray, WithElementType(entity.FieldTypeVarChar), WithMaxLength(TestMaxLen), WithMaxCapacity(capacity)),
	}
	return fields
}

// GenAllFields gen fields with all scala field types
func GenAllFields() []*entity.Field {
	allFields := []*entity.Field{
		GenField(DefaultIntFieldName, entity.FieldTypeInt64, WithIsPrimaryKey(true)),               // int64
		GenField(DefaultBoolFieldName, entity.FieldTypeBool),                                       // bool
		GenField(DefaultInt8FieldName, entity.FieldTypeInt8),                                       // int8
		GenField(DefaultInt16FieldName, entity.FieldTypeInt16),                                     // int16
		GenField(DefaultInt32FieldName, entity.FieldTypeInt32),                                     // int32
		GenField(DefaultFloatFieldName, entity.FieldTypeFloat),                                     // float
		GenField(DefaultDoubleFieldName, entity.FieldTypeDouble),                                   // double
		GenField(DefaultVarcharFieldName, entity.FieldTypeVarChar, WithMaxLength(MaxLength)),       // varchar
		GenField(DefaultJSONFieldName, entity.FieldTypeJSON),                                       // json
		GenField(DefaultFloatVecFieldName, entity.FieldTypeFloatVector, WithDim(DefaultDim)),       // float vector
		GenField(DefaultFloat16VecFieldName, entity.FieldTypeFloat16Vector, WithDim(DefaultDim)),   // float16 vector
		GenField(DefaultBFloat16VecFieldName, entity.FieldTypeBFloat16Vector, WithDim(DefaultDim)), // bf16 vector
		GenField(DefaultBinaryVecFieldName, entity.FieldTypeBinaryVector, WithDim(DefaultDim)),     // binary vector
	}
	allFields = append(allFields, GenAllArrayFields()...)
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

	case entity.FieldTypeArray:
		return GenArrayColumnData(start, nb, fieldName, opts...)

	case entity.FieldTypeFloatVector:
		vecFloatValues := make([][]float32, 0, nb)
		for i := start; i < start+nb; i++ {
			vec := GenFloatVector(opt.dim)
			vecFloatValues = append(vecFloatValues, vec)
		}
		return entity.NewColumnFloatVector(fieldName, int(opt.dim), vecFloatValues)
	case entity.FieldTypeBinaryVector:
		binaryVectors := make([][]byte, 0, nb)
		for i := 0; i < nb; i++ {
			vec := GenBinaryVector(opt.dim)
			binaryVectors = append(binaryVectors, vec)
		}
		return entity.NewColumnBinaryVector(fieldName, int(opt.dim), binaryVectors)
	case entity.FieldTypeFloat16Vector:
		fp16Vectors := make([][]byte, 0, nb)
		for i := start; i < start+nb; i++ {
			vec := GenFloat16Vector(opt.dim)
			fp16Vectors = append(fp16Vectors, vec)
		}
		return entity.NewColumnFloat16Vector(fieldName, int(opt.dim), fp16Vectors)
	case entity.FieldTypeBFloat16Vector:
		bf16Vectors := make([][]byte, 0, nb)
		for i := start; i < start+nb; i++ {
			vec := GenBFloat16Vector(opt.dim)
			bf16Vectors = append(bf16Vectors, vec)
		}
		return entity.NewColumnBFloat16Vector(fieldName, int(opt.dim), bf16Vectors)
	default:
		return nil
	}
}

func GenArrayColumnData(start int, nb int, fieldName string, opts ...GenColumnDataOption) entity.Column {
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}
	eleType := opt.ElementType
	capacity := int(opt.capacity)
	switch eleType {
	case entity.FieldTypeBool:
		boolValues := make([][]bool, 0, nb)
		for i := start; i < start+nb; i++ {
			boolArray := make([]bool, 0, capacity)
			for j := 0; j < capacity; j++ {
				boolArray = append(boolArray, i%2 == 0)
			}
			boolValues = append(boolValues, boolArray)
		}
		return entity.NewColumnBoolArray(fieldName, boolValues)
	case entity.FieldTypeInt8:
		int8Values := make([][]int8, 0, nb)
		for i := start; i < start+nb; i++ {
			int8Array := make([]int8, 0, capacity)
			for j := 0; j < capacity; j++ {
				int8Array = append(int8Array, int8(i+j))
			}
			int8Values = append(int8Values, int8Array)
		}
		return entity.NewColumnInt8Array(fieldName, int8Values)
	case entity.FieldTypeInt16:
		int16Values := make([][]int16, 0, nb)
		for i := start; i < start+nb; i++ {
			int16Array := make([]int16, 0, capacity)
			for j := 0; j < capacity; j++ {
				int16Array = append(int16Array, int16(i+j))
			}
			int16Values = append(int16Values, int16Array)
		}
		return entity.NewColumnInt16Array(fieldName, int16Values)
	case entity.FieldTypeInt32:
		int32Values := make([][]int32, 0, nb)
		for i := start; i < start+nb; i++ {
			int32Array := make([]int32, 0, capacity)
			for j := 0; j < capacity; j++ {
				int32Array = append(int32Array, int32(i+j))
			}
			int32Values = append(int32Values, int32Array)
		}
		return entity.NewColumnInt32Array(fieldName, int32Values)
	case entity.FieldTypeInt64:
		int64Values := make([][]int64, 0, nb)
		for i := start; i < start+nb; i++ {
			int64Array := make([]int64, 0, capacity)
			for j := 0; j < capacity; j++ {
				int64Array = append(int64Array, int64(i+j))
			}
			int64Values = append(int64Values, int64Array)
		}
		return entity.NewColumnInt64Array(fieldName, int64Values)
	case entity.FieldTypeFloat:
		floatValues := make([][]float32, 0, nb)
		for i := start; i < start+nb; i++ {
			floatArray := make([]float32, 0, capacity)
			for j := 0; j < capacity; j++ {
				floatArray = append(floatArray, float32(i+j))
			}
			floatValues = append(floatValues, floatArray)
		}
		return entity.NewColumnFloatArray(fieldName, floatValues)
	case entity.FieldTypeDouble:
		doubleValues := make([][]float64, 0, nb)
		for i := start; i < start+nb; i++ {
			doubleArray := make([]float64, 0, capacity)
			for j := 0; j < capacity; j++ {
				doubleArray = append(doubleArray, float64(i+j))
			}
			doubleValues = append(doubleValues, doubleArray)
		}
		return entity.NewColumnDoubleArray(fieldName, doubleValues)
	case entity.FieldTypeVarChar:
		varcharValues := make([][][]byte, 0, nb)
		for i := start; i < start+nb; i++ {
			varcharArray := make([][]byte, 0, capacity)
			for j := 0; j < capacity; j++ {
				var buf bytes.Buffer
				buf.WriteString(strconv.Itoa(i + j))
				varcharArray = append(varcharArray, buf.Bytes())
			}
			varcharValues = append(varcharValues, varcharArray)
		}
		return entity.NewColumnVarCharArray(fieldName, varcharValues)
	default:
		return nil
	}
}

type JSONStruct struct {
	Number int32   `json:"number" milvus:"name:number"`
	String string  `json:"string" milvus:"name:string"`
	Bool   bool    `json:"bool" milvus:"name:bool"`
	List   []int64 `json:"list" milvus:"name:list"`
}

// GenDefaultJSONData gen default column with data
func GenDefaultJSONData(columnName string, start int, nb int) *entity.ColumnJSONBytes {
	jsonValues := make([][]byte, 0, nb)
	var m interface{}
	for i := start; i < start+nb; i++ {
		// kv value
		if i < (start+nb)/2 {
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
		} else {
			// int, float, string, list
			switch i % 4 {
			case 0:
				m = i
			case 1:
				m = float32(i)
			case 2:
				m = strconv.Itoa(i)
			case 3:
				m = []int64{int64(i), int64(i + 1)}
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

func GenAllArrayData(start int, nb int, opts ...GenColumnDataOption) []entity.Column {
	// how to pass different capacity for different column
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}
	data := []entity.Column{
		GenArrayColumnData(start, nb, DefaultBoolArrayField, WithArrayElementType(entity.FieldTypeBool), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultInt8ArrayField, WithArrayElementType(entity.FieldTypeInt8), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultInt16ArrayField, WithArrayElementType(entity.FieldTypeInt16), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultInt32ArrayField, WithArrayElementType(entity.FieldTypeInt32), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultInt64ArrayField, WithArrayElementType(entity.FieldTypeInt64), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultFloatArrayField, WithArrayElementType(entity.FieldTypeFloat), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultDoubleArrayField, WithArrayElementType(entity.FieldTypeDouble), WithArrayCapacity(opt.capacity)),
		GenArrayColumnData(start, nb, DefaultVarcharArrayField, WithArrayElementType(entity.FieldTypeVarChar), WithArrayCapacity(opt.capacity)),
	}
	return data
}

func GenAllVectorsData(start int, nb int, dim int64, opts ...GenColumnDataOption) []entity.Column {
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}

	// prepare data
	data := []entity.Column{
		GenColumnData(start, nb, entity.FieldTypeInt64, "int64"),
		GenColumnData(start, nb, entity.FieldTypeFloatVector, "floatVec", WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeFloat16Vector, "fp16Vec", WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeBFloat16Vector, "bf16Vec", WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeBinaryVector, "binaryVec", WithVectorDim(dim)),
	}
	return data
}

func GenAllFieldsData(start int, nb int, dim int64, opts ...GenColumnDataOption) []entity.Column {
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}
	// prepare data
	data := []entity.Column{
		GenColumnData(start, nb, entity.FieldTypeInt64, DefaultIntFieldName),
		GenColumnData(start, nb, entity.FieldTypeBool, DefaultBoolFieldName),
		GenColumnData(start, nb, entity.FieldTypeInt8, DefaultInt8FieldName),
		GenColumnData(start, nb, entity.FieldTypeInt16, DefaultInt16FieldName),
		GenColumnData(start, nb, entity.FieldTypeInt32, DefaultInt32FieldName),
		GenColumnData(start, nb, entity.FieldTypeFloat, DefaultFloatFieldName),
		GenColumnData(start, nb, entity.FieldTypeDouble, DefaultDoubleFieldName),
		GenColumnData(start, nb, entity.FieldTypeVarChar, DefaultVarcharFieldName),
		GenDefaultJSONData(DefaultJSONFieldName, start, nb),
		GenColumnData(start, nb, entity.FieldTypeFloatVector, DefaultFloatVecFieldName, WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeFloat16Vector, DefaultFloat16VecFieldName, WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeBFloat16Vector, DefaultBFloat16VecFieldName, WithVectorDim(dim)),
		GenColumnData(start, nb, entity.FieldTypeBinaryVector, DefaultBinaryVecFieldName, WithVectorDim(dim)),
	}
	data = append(data, GenAllArrayData(start, nb, opts...)...)
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

type Array struct {
	BoolArray    []bool    `json:"boolArray" milvus:"name:boolArray"`
	Int8Array    []int8    `json:"int8Array" milvus:"name:int8Array"`
	Int16Array   []int16   `json:"int16Array" milvus:"name:int16Array"`
	Int32Array   []int32   `json:"int32Array" milvus:"name:int32Array"`
	Int64Array   []int64   `json:"int64Array" milvus:"name:int64Array"`
	FloatArray   []float32 `json:"floatArray" milvus:"name:floatArray"`
	DoubleArray  []float64 `json:"doubleArray" milvus:"name:doubleArray"`
	VarcharArray [][]byte  `json:"varcharArray" milvus:"name:varcharArray"`
}

func GenDefaultRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
	}

	type DynamicRow struct {
		Int64    int64     `json:"int64" milvus:"name:int64"`
		Float    float32   `json:"float" milvus:"name:float"`
		FloatVec []float32 `json:"floatVec" milvus:"name:floatVec"`
		Dynamic  Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		baseRow := BaseRow{
			Int64:    int64(i),
			Float:    float32(i),
			FloatVec: GenFloatVector(dim),
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
				Int64:    baseRow.Int64,
				Float:    baseRow.Float,
				FloatVec: baseRow.FloatVec,
				Dynamic:  dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenDefaultBinaryRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64     int64   `json:"int64" milvus:"name:int64"`
		Float     float32 `json:"float" milvus:"name:float"`
		BinaryVec []byte  `json:"binaryVec" milvus:"name:binaryVec"`
	}

	type DynamicRow struct {
		Int64     int64   `json:"int64" milvus:"name:int64"`
		Float     float32 `json:"float" milvus:"name:float"`
		BinaryVec []byte  `json:"binaryVec" milvus:"name:binaryVec"`
		Dynamic   Dynamic `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		baseRow := BaseRow{
			Int64:     int64(i),
			Float:     float32(i),
			BinaryVec: GenBinaryVector(dim),
		}
		if enableDynamicField {
			dynamic := Dynamic{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}

			dynamicRow := DynamicRow{
				Int64:     baseRow.Int64,
				Float:     baseRow.Float,
				BinaryVec: baseRow.BinaryVec,
				Dynamic:   dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenDefaultVarcharRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Varchar   string `json:"varchar" milvus:"name:varchar"`
		BinaryVec []byte `json:"binaryVec" milvus:"name:binaryVec"`
	}

	type DynamicRow struct {
		Varchar   string  `json:"varchar" milvus:"name:varchar"`
		BinaryVec []byte  `json:"binaryVec" milvus:"name:binaryVec"`
		Dynamic   Dynamic `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		baseRow := BaseRow{
			Varchar:   strconv.Itoa(i),
			BinaryVec: GenBinaryVector(dim),
		}

		if enableDynamicField {
			dynamic := Dynamic{
				Number: int32(i),
				String: strconv.Itoa(i),
				Bool:   i%2 == 0,
				List:   []int64{int64(i), int64(i + 1)},
			}

			dynamicRow := DynamicRow{
				Varchar:   baseRow.Varchar,
				BinaryVec: baseRow.BinaryVec,
				Dynamic:   dynamic,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenDefaultJSONRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64    int64      `json:"int64" milvus:"name:int64"`
		Float    float32    `json:"float" milvus:"name:float"`
		FloatVec []float32  `json:"floatVec" milvus:"name:floatVec"`
		JSON     JSONStruct `json:"json" milvus:"name:json"`
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

	for i := start; i < start+nb; i++ {
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
		// base row
		baseRow := BaseRow{
			Int64:    int64(i),
			Float:    float32(i),
			FloatVec: GenFloatVector(dim),
			JSON:     jsonStruct,
		}
		if enableDynamicField {
			baseDynamicRow := BaseDynamicRow{
				Int64:    baseRow.Int64,
				Float:    baseRow.Float,
				FloatVec: baseRow.FloatVec,
				JSON:     baseRow.JSON,
				Number:   int32(i),
				String:   strconv.Itoa(i),
				Bool:     i%2 == 0,
				//List:     []int64{int64(i), int64(i + 1)},
			}

			rows = append(rows, &baseDynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenAllArrayRow(index int, opts ...GenColumnDataOption) Array {
	opt := &genDataOpt{}
	for _, o := range opts {
		o(opt)
	}
	var capacity int
	if opt.capacity != 0 {
		capacity = int(opt.capacity)
	} else {
		capacity = TestCapacity
	}

	boolRow := make([]bool, 0, capacity)
	int8Row := make([]int8, 0, capacity)
	int16Row := make([]int16, 0, capacity)
	int32Row := make([]int32, 0, capacity)
	int64Row := make([]int64, 0, capacity)
	floatRow := make([]float32, 0, capacity)
	doubleRow := make([]float64, 0, capacity)
	varcharRow := make([][]byte, 0, capacity)
	for j := 0; j < capacity; j++ {
		boolRow = append(boolRow, index%2 == 0)
		int8Row = append(int8Row, int8(index+j))
		int16Row = append(int16Row, int16(index+j))
		int32Row = append(int32Row, int32(index+j))
		int64Row = append(int64Row, int64(index+j))
		floatRow = append(floatRow, float32(index+j))
		doubleRow = append(doubleRow, float64(index+j))
		var buf bytes.Buffer
		buf.WriteString(strconv.Itoa(index + j))
		varcharRow = append(varcharRow, buf.Bytes())
	}
	arrayRow := Array{
		BoolArray:    boolRow,
		Int8Array:    int8Row,
		Int16Array:   int16Row,
		Int32Array:   int32Row,
		Int64Array:   int64Row,
		FloatArray:   floatRow,
		DoubleArray:  doubleRow,
		VarcharArray: varcharRow,
	}
	return arrayRow
}

func GenDefaultArrayRows(start int, nb int, dim int64, enableDynamicField bool, opts ...GenColumnDataOption) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64        int64     `json:"int64" milvus:"name:int64"`
		Float        float32   `json:"float" milvus:"name:float"`
		FloatVec     []float32 `json:"floatVec" milvus:"name:floatVec"`
		BoolArray    []bool    `json:"boolArray" milvus:"name:boolArray"`
		Int8Array    []int8    `json:"int8Array" milvus:"name:int8Array"`
		Int16Array   []int16   `json:"int16Array" milvus:"name:int16Array"`
		Int32Array   []int32   `json:"int32Array" milvus:"name:int32Array"`
		Int64Array   []int64   `json:"int64Array" milvus:"name:int64Array"`
		FloatArray   []float32 `json:"floatArray" milvus:"name:floatArray"`
		DoubleArray  []float64 `json:"doubleArray" milvus:"name:doubleArray"`
		VarcharArray [][]byte  `json:"varcharArray" milvus:"name:varcharArray"`
	}

	type DynamicRow struct {
		Int64        int64     `json:"int64" milvus:"name:int64"`
		Float        float32   `json:"float" milvus:"name:float"`
		FloatVec     []float32 `json:"floatVec" milvus:"name:floatVec"`
		BoolArray    []bool    `json:"boolArray" milvus:"name:boolArray"`
		Int8Array    []int8    `json:"int8Array" milvus:"name:int8Array"`
		Int16Array   []int16   `json:"int16Array" milvus:"name:int16Array"`
		Int32Array   []int32   `json:"int32Array" milvus:"name:int32Array"`
		Int64Array   []int64   `json:"int64Array" milvus:"name:int64Array"`
		FloatArray   []float32 `json:"floatArray" milvus:"name:floatArray"`
		DoubleArray  []float64 `json:"doubleArray" milvus:"name:doubleArray"`
		VarcharArray [][]byte  `json:"varcharArray" milvus:"name:varcharArray"`
		Dynamic      Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		// json and dynamic field
		dynamicJSON := Dynamic{
			Number: int32(i),
			String: strconv.Itoa(i),
			Bool:   i%2 == 0,
			List:   []int64{int64(i), int64(i + 1)},
		}
		arrayRow := GenAllArrayRow(i, opts...)
		baseRow := BaseRow{
			Int64:        int64(i),
			Float:        float32(i),
			FloatVec:     GenFloatVector(dim),
			BoolArray:    arrayRow.BoolArray,
			Int8Array:    arrayRow.Int8Array,
			Int16Array:   arrayRow.Int16Array,
			Int32Array:   arrayRow.Int32Array,
			Int64Array:   arrayRow.Int64Array,
			FloatArray:   arrayRow.FloatArray,
			DoubleArray:  arrayRow.DoubleArray,
			VarcharArray: arrayRow.VarcharArray,
		}
		if enableDynamicField {
			dynamicRow := DynamicRow{
				Int64:        baseRow.Int64,
				Float:        baseRow.Float,
				FloatVec:     baseRow.FloatVec,
				BoolArray:    arrayRow.BoolArray,
				Int8Array:    arrayRow.Int8Array,
				Int16Array:   arrayRow.Int16Array,
				Int32Array:   arrayRow.Int32Array,
				Int64Array:   arrayRow.Int64Array,
				FloatArray:   arrayRow.FloatArray,
				DoubleArray:  arrayRow.DoubleArray,
				VarcharArray: arrayRow.VarcharArray,
				Dynamic:      dynamicJSON,
			}

			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenAllVectorsRows(start int, nb int, dim int64, enableDynamicField bool) []interface{} {
	rows := make([]interface{}, 0, nb)
	type BaseRow struct {
		Int64       int64     `json:"int64" milvus:"name:int64"`
		FloatVec    []float32 `json:"floatVec" milvus:"name:floatVec"`
		Float16Vec  []byte    `json:"fp16Vec" milvus:"name:fp16Vec"`
		BFloat16Vec []byte    `json:"bf16Vec" milvus:"name:bf16Vec"`
		BinaryVec   []byte    `json:"binaryVec" milvus:"name:binaryVec"`
	}

	type DynamicRow struct {
		Int64       int64     `json:"int64" milvus:"name:int64"`
		FloatVec    []float32 `json:"floatVec" milvus:"name:floatVec"`
		Float16Vec  []byte    `json:"fp16Vec" milvus:"name:fp16Vec"`
		BFloat16Vec []byte    `json:"bf16Vec" milvus:"name:bf16Vec"`
		BinaryVec   []byte    `json:"binaryVec" milvus:"name:binaryVec"`
		Dynamic     Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		baseRow := BaseRow{
			Int64:       int64(i),
			FloatVec:    GenFloatVector(dim),
			Float16Vec:  GenFloat16Vector(dim),
			BFloat16Vec: GenBFloat16Vector(dim),
			BinaryVec:   GenBinaryVector(dim),
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
				Int64:       baseRow.Int64,
				FloatVec:    baseRow.FloatVec,
				Float16Vec:  baseRow.Float16Vec,
				BFloat16Vec: baseRow.BFloat16Vec,
				BinaryVec:   baseRow.BinaryVec,
				Dynamic:     dynamicJSON,
			}
			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
		}
	}
	return rows
}

func GenAllFieldsRows(start int, nb int, dim int64, enableDynamicField bool, opts ...GenColumnDataOption) []interface{} {
	rows := make([]interface{}, 0, nb)

	// BaseRow generate insert rows
	type BaseRow struct {
		Int64        int64     `json:"int64" milvus:"name:int64"`
		Bool         bool      `json:"bool" milvus:"name:bool"`
		Int8         int8      `json:"int8" milvus:"name:int8"`
		Int16        int16     `json:"int16" milvus:"name:int16"`
		Int32        int32     `json:"int32" milvus:"name:int32"`
		Float        float32   `json:"float" milvus:"name:float"`
		Double       float64   `json:"double" milvus:"name:double"`
		Varchar      string    `json:"varchar" milvus:"name:varchar"`
		JSON         Dynamic   `json:"json" milvus:"name:json"`
		FloatVec     []float32 `json:"floatVec" milvus:"name:floatVec"`
		Float16Vec   []byte    `json:"fp16Vec" milvus:"name:fp16Vec"`
		BFloat16Vec  []byte    `json:"bf16Vec" milvus:"name:bf16Vec"`
		BinaryVec    []byte    `json:"binaryVec" milvus:"name:binaryVec"`
		BoolArray    []bool    `json:"boolArray" milvus:"name:boolArray"`
		Int8Array    []int8    `json:"int8Array" milvus:"name:int8Array"`
		Int16Array   []int16   `json:"int16Array" milvus:"name:int16Array"`
		Int32Array   []int32   `json:"int32Array" milvus:"name:int32Array"`
		Int64Array   []int64   `json:"int64Array" milvus:"name:int64Array"`
		FloatArray   []float32 `json:"floatArray" milvus:"name:floatArray"`
		DoubleArray  []float64 `json:"doubleArray" milvus:"name:doubleArray"`
		VarcharArray [][]byte  `json:"varcharArray" milvus:"name:varcharArray"`
	}

	type DynamicRow struct {
		Int64        int64     `json:"int64" milvus:"name:int64"`
		Bool         bool      `json:"bool" milvus:"name:bool"`
		Int8         int8      `json:"int8" milvus:"name:int8"`
		Int16        int16     `json:"int16" milvus:"name:int16"`
		Int32        int32     `json:"int32" milvus:"name:int32"`
		Float        float32   `json:"float" milvus:"name:float"`
		Double       float64   `json:"double" milvus:"name:double"`
		Varchar      string    `json:"varchar" milvus:"name:varchar"`
		JSON         Dynamic   `json:"json" milvus:"name:json"`
		FloatVec     []float32 `json:"floatVec" milvus:"name:floatVec"`
		Float16Vec   []byte    `json:"fp16Vec" milvus:"name:fp16Vec"`
		BFloat16Vec  []byte    `json:"bf16Vec" milvus:"name:bf16Vec"`
		BinaryVec    []byte    `json:"binaryVec" milvus:"name:binaryVec"`
		BoolArray    []bool    `json:"boolArray" milvus:"name:boolArray"`
		Int8Array    []int8    `json:"int8Array" milvus:"name:int8Array"`
		Int16Array   []int16   `json:"int16Array" milvus:"name:int16Array"`
		Int32Array   []int32   `json:"int32Array" milvus:"name:int32Array"`
		Int64Array   []int64   `json:"int64Array" milvus:"name:int64Array"`
		FloatArray   []float32 `json:"floatArray" milvus:"name:floatArray"`
		DoubleArray  []float64 `json:"doubleArray" milvus:"name:doubleArray"`
		VarcharArray [][]byte  `json:"varcharArray" milvus:"name:varcharArray"`
		Dynamic      Dynamic   `json:"dynamic" milvus:"name:dynamic"`
	}

	for i := start; i < start+nb; i++ {
		arrayRow := GenAllArrayRow(i, opts...)
		baseRow := BaseRow{
			Int64:        int64(i),
			Bool:         i%2 == 0,
			Int8:         int8(i),
			Int16:        int16(i),
			Int32:        int32(i),
			Float:        float32(i),
			Double:       float64(i),
			Varchar:      strconv.Itoa(i),
			FloatVec:     GenFloatVector(dim),
			Float16Vec:   GenFloat16Vector(dim),
			BFloat16Vec:  GenBFloat16Vector(dim),
			BinaryVec:    GenBinaryVector(dim),
			BoolArray:    arrayRow.BoolArray,
			Int8Array:    arrayRow.Int8Array,
			Int16Array:   arrayRow.Int16Array,
			Int32Array:   arrayRow.Int32Array,
			Int64Array:   arrayRow.Int64Array,
			FloatArray:   arrayRow.FloatArray,
			DoubleArray:  arrayRow.DoubleArray,
			VarcharArray: arrayRow.VarcharArray,
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
				Int64:        baseRow.Int64,
				Bool:         baseRow.Bool,
				Int8:         baseRow.Int8,
				Int16:        baseRow.Int16,
				Int32:        baseRow.Int32,
				Float:        baseRow.Float,
				Double:       baseRow.Double,
				Varchar:      baseRow.Varchar,
				FloatVec:     baseRow.FloatVec,
				Float16Vec:   baseRow.Float16Vec,
				BFloat16Vec:  baseRow.BFloat16Vec,
				BinaryVec:    baseRow.BinaryVec,
				BoolArray:    arrayRow.BoolArray,
				Int8Array:    arrayRow.Int8Array,
				Int16Array:   arrayRow.Int16Array,
				Int32Array:   arrayRow.Int32Array,
				Int64Array:   arrayRow.Int64Array,
				FloatArray:   arrayRow.FloatArray,
				DoubleArray:  arrayRow.DoubleArray,
				VarcharArray: arrayRow.VarcharArray,
				Dynamic:      dynamicJSON,
			}
			rows = append(rows, dynamicRow)
		} else {
			rows = append(rows, &baseRow)
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
var SupportFloatMetricType = []entity.MetricType{
	entity.L2,
	entity.IP,
	entity.COSINE,
}

var SupportBinFlatMetricType = []entity.MetricType{
	entity.JACCARD,
	entity.HAMMING,
	entity.SUBSTRUCTURE,
	entity.SUPERSTRUCTURE,
}

var SupportBinIvfFlatMetricType = []entity.MetricType{
	entity.JACCARD,
	entity.HAMMING,
}

// GenAllFloatIndex gen all float vector index
func GenAllFloatIndex() []entity.Index {
	nlist := 128
	var allFloatIndex []entity.Index
	for _, metricType := range SupportFloatMetricType {
		idxFlat, _ := entity.NewIndexFlat(metricType)
		idxIvfFlat, _ := entity.NewIndexIvfFlat(metricType, nlist)
		idxIvfSq8, _ := entity.NewIndexIvfSQ8(metricType, nlist)
		idxIvfPq, _ := entity.NewIndexIvfPQ(metricType, nlist, 16, 8)
		idxHnsw, _ := entity.NewIndexHNSW(metricType, 8, 96)
		idxScann, _ := entity.NewIndexSCANN(metricType, 16, false)
		// TODO waiting for PR https://github.com/milvus-io/milvus/pull/30716
		//idxDiskAnn, _ := entity.NewIndexDISKANN(metricType)
		allFloatIndex = append(allFloatIndex, idxFlat, idxIvfFlat, idxIvfSq8, idxIvfPq, idxHnsw, idxScann)
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
			vector := GenFloatVector(dim)
			vectors = append(vectors, entity.FloatVector(vector))
		}
	case entity.FieldTypeBinaryVector:
		for i := 0; i < nq; i++ {
			vector := GenBinaryVector(dim)
			vectors = append(vectors, entity.BinaryVector(vector))
		}
	case entity.FieldTypeFloat16Vector:
		for i := 0; i < nq; i++ {
			vector := GenFloat16Vector(dim)
			vectors = append(vectors, entity.Float16Vector(vector))
		}
	case entity.FieldTypeBFloat16Vector:
		for i := 0; i < nq; i++ {
			vector := GenBFloat16Vector(dim)
			vectors = append(vectors, entity.BFloat16Vector(vector))
		}
	}
	return vectors
}

// InvalidExprStruct invalid expr
type InvalidExprStruct struct {
	Expr   string
	ErrNil bool
	ErrMsg string
}

var InvalidExpressions = []InvalidExprStruct{
	{Expr: "id in [0]", ErrNil: true, ErrMsg: "fieldName(id) not found"},                                          // not exist field but no error
	{Expr: "int64 in not [0]", ErrNil: false, ErrMsg: "cannot parse expression"},                                  // wrong term expr keyword
	{Expr: "int64 < floatVec", ErrNil: false, ErrMsg: "not supported"},                                            // unsupported compare field
	{Expr: "floatVec in [0]", ErrNil: false, ErrMsg: "cannot be casted to FloatVector"},                           // value and field type mismatch
	{Expr: fmt.Sprintf("%s == 1", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},                                // hist empty
	{Expr: fmt.Sprintf("%s like 'a%%' ", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},                         // hist empty
	{Expr: fmt.Sprintf("%s like `a%%` ", DefaultJSONFieldName), ErrNil: false, ErrMsg: "cannot parse expression"}, // ``
	{Expr: fmt.Sprintf("%s > 1", DefaultDynamicFieldName), ErrNil: true, ErrMsg: ""},                              // hits empty
	{Expr: fmt.Sprintf("%s[\"dynamicList\"] == [2, 3]", DefaultDynamicFieldName), ErrNil: true, ErrMsg: ""},
	{Expr: fmt.Sprintf("%s['a'] == [2, 3]", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},      // json field not exist
	{Expr: fmt.Sprintf("%s['number'] == [2, 3]", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""}, // field exist but type not match
	{Expr: fmt.Sprintf("%s[0] == [2, 3]", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},        // field exist but type not match
	{Expr: fmt.Sprintf("json_contains (%s['number'], 2)", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},
	{Expr: fmt.Sprintf("json_contains (%s['list'], [2])", DefaultJSONFieldName), ErrNil: true, ErrMsg: ""},
	{Expr: fmt.Sprintf("json_contains_all (%s['list'], 2)", DefaultJSONFieldName), ErrNil: false, ErrMsg: "contains_all operation element must be an array"},
	{Expr: fmt.Sprintf("JSON_CONTAINS_ANY (%s['list'], 2)", DefaultJSONFieldName), ErrNil: false, ErrMsg: "contains_any operation element must be an array"},
	{Expr: fmt.Sprintf("json_contains_aby (%s['list'], 2)", DefaultJSONFieldName), ErrNil: false, ErrMsg: "invalid expression: json_contains_aby"},
	{Expr: fmt.Sprintf("json_contains_aby (%s['list'], 2)", DefaultJSONFieldName), ErrNil: false, ErrMsg: "invalid expression: json_contains_aby"},
	{Expr: fmt.Sprintf("%s[-1] > %d", DefaultInt8ArrayField, TestCapacity), ErrNil: false, ErrMsg: "cannot parse expression"}, //  array[-1] >
	{Expr: fmt.Sprintf(fmt.Sprintf("%s[-1] > 1", DefaultJSONFieldName)), ErrNil: false, ErrMsg: "invalid expression"},         //  json[-1] >
}

// --- search utils ---
