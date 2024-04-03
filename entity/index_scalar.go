package entity

var _ Index = (*indexScalar)(nil)

type indexScalar struct {
	baseIndex
}

func (i *indexScalar) Params() map[string]string {
	result := make(map[string]string)
	if i.baseIndex.it != "" {
		result[tIndexType] = string(i.baseIndex.it)
	}
	return result
}

func NewScalarIndex() Index {
	return &indexScalar{
		baseIndex: baseIndex{},
	}
}

func NewScalarIndexWithType(indexType IndexType) Index {
	return &indexScalar{
		baseIndex: baseIndex{
			it: indexType,
		},
	}
}
