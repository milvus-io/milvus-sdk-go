package milvus

import (
	"testing"

	"github.com/tidwall/sjson"
)

func TestParams_Set(t *testing.T) {
	s := NewParams("").Set("foo", "bar").GetParams()
	s1, _ := sjson.Set("", "foo", "bar")
	if s != s1 {
		t.Errorf("%s should be equal to %s", s, s1)
	}

	s = NewParams("").Set("foo", "bar").Set("foo.bar", "foo.bar").GetParams()
	s1, _ = sjson.Set("", "foo", "bar")
	s1, _ = sjson.Set(s1, "foo.bar", "foo.bar")
	if s != s1 {
		t.Errorf("%s should be equal to %s", s, s1)
	}
}
