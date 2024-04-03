package entity

var _ Index = (*indexScalar)(nil)

type indexScalar struct {
	baseIndex
}

func (i *indexScalar) Params() map[string]string {
	return map[string]string{
		tIndexType: string(i.baseIndex.it),
	}
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
