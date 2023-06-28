package client

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestClientConfig(t *testing.T) {
	// empty config.
	assertConfig(
		t,
		&Config{},
		"", "", false, true,
	)
	// local host
	assertConfig(
		t,
		&Config{
			Address: "localhost:19540",
		},
		"localhost:19540", "", false, false,
	)
	// local secure host
	assertConfig(
		t,
		&Config{
			Address:       "localhost:19540",
			EnableTLSAuth: true,
		},
		"localhost:19540", "", true, false,
	)
	// local http host
	assertConfig(
		t,
		&Config{
			Address: "http://localhost:19540",
		},
		"localhost:19540", "", false, false,
	)
	// local https host
	assertConfig(
		t,
		&Config{
			Address: "https://localhost:19540",
		},
		"localhost:19540", "", true, false,
	)
	// remote https host
	assertConfig(
		t,
		&Config{
			Address: "https://xxxx-xxxx-xxxxx.com",
		},
		"xxxx-xxxx-xxxxx.com", "", true, false,
	)
	// remote https host with dbname
	assertConfig(
		t,
		&Config{
			Address: "https://xxxx-xxxxxxxxxxxxxx-xxxxxx.aws-us-west-2.vectordb-sit.zillizcloud.com:19530/database_name",
			APIKey:  "test-token",
		},
		"xxxx-xxxxxxxxxxxxxx-xxxxxx.aws-us-west-2.vectordb-sit.zillizcloud.com:19530", "database_name", true, false,
	)

	// local secure host
	assertConfig(
		t,
		&Config{
			Address: "https://localhost:19540",
		},
		"localhost:19540", "", true, false,
	)

	assertConfig(
		t,
		&Config{
			Address: "https://localhost:8080",
		},
		"localhost:8080", "", true, false,
	)
	assertConfig(
		t,
		&Config{
			Address: "https://localhost:port",
		},
		"", "", false, true,
	)
	assertConfig(
		t,
		&Config{
			Address: "http://localhost:8080",
		},
		"localhost:8080", "", false, false,
	)
	assertConfig(
		t,
		&Config{
			Address: "http://localhost:port",
		},
		"", "", false, true,
	)
	assertConfig(
		t,
		&Config{
			Address: "localhost:8080",
		},
		"localhost:8080", "", false, false,
	)
	assertConfig(
		t,
		&Config{
			Address: "localhost:8080",
			DBName:  "test_db",
		},
		"localhost:8080", "test_db", false, false,
	)
}

func assertConfig(t *testing.T, c *Config, host string, db string, secure bool, isErr bool) {
	err := c.parse()
	if isErr {
		assert.NotNil(t, err)
		return
	}
	assert.Nil(t, err)
	assert.Equal(t, c.EnableTLSAuth, secure)
	assert.Equal(t, c.DBName, db)
	assert.Equal(t, c.getParsedAddress(), host)
}
