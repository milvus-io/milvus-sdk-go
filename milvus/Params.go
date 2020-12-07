package milvus

import (
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type Params struct {
	params string
}

func NewParams(params string) *Params {
	return &Params{params}
}

func (p *Params) Get(key string) gjson.Result {
	return gjson.Get(p.params, key)
}

func (p *Params) Set(key string, value interface{}) *Params {
	res, _ := sjson.Set(p.params, key, value)
	p.params = res
	return p
}

func (p *Params) String() string {
	return p.params
}
