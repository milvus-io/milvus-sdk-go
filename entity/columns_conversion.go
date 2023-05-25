package entity

func (c *ColumnInt8) GetAsInt64(idx int) (int64, error) {
	v, err := c.ValueByIdx(idx)
	return int64(v), err
}

func (c *ColumnInt16) GetAsInt64(idx int) (int64, error) {
	v, err := c.ValueByIdx(idx)
	return int64(v), err
}

func (c *ColumnInt32) GetAsInt64(idx int) (int64, error) {
	v, err := c.ValueByIdx(idx)
	return int64(v), err
}

func (c *ColumnInt64) GetAsInt64(idx int) (int64, error) {
	return c.ValueByIdx(idx)
}

func (c *ColumnString) GetAsString(idx int) (string, error) {
	return c.ValueByIdx(idx)
}

func (c *ColumnFloat) GetAsDouble(idx int) (float64, error) {
	v, err := c.ValueByIdx(idx)
	return float64(v), err
}

func (c *ColumnDouble) GetAsDouble(idx int) (float64, error) {
	return c.ValueByIdx(idx)
}

func (c *ColumnBool) GetAsBool(idx int) (bool, error) {
	return c.ValueByIdx(idx)
}
