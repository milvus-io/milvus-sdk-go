package entity

var _ Index = (*indexScalar)(nil)

type indexScalar struct {
	baseIndex
}

func (i *indexScalar) Params() map[string]string {
	return map[string]string{}
}

func NewScalarIndex() Index {
	return &indexScalar{
		baseIndex: baseIndex{
			it: Scalar,
		},
	}
}
