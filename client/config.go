package client

import (
	"context"
	"crypto/tls"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	grpc_retry "github.com/grpc-ecosystem/go-grpc-middleware/retry"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

var regexValidScheme = regexp.MustCompile(`^https?:\/\/`)

// DefaultGrpcOpts is GRPC options for milvus client.
var DefaultGrpcOpts = []grpc.DialOption{
	grpc.WithBlock(),
	grpc.WithKeepaliveParams(keepalive.ClientParameters{
		Time:                5 * time.Second,
		Timeout:             10 * time.Second,
		PermitWithoutStream: true,
	}),
	grpc.WithConnectParams(grpc.ConnectParams{
		Backoff: backoff.Config{
			BaseDelay:  100 * time.Millisecond,
			Multiplier: 1.6,
			Jitter:     0.2,
			MaxDelay:   3 * time.Second,
		},
		MinConnectTimeout: 3 * time.Second,
	}),
}

// Config for milvus client.
type Config struct {
	Address       string // Remote address, "localhost:19530".
	Username      string // Username for auth.
	Password      string // Password for auth.
	DBName        string // DBName for this client.
	EnableTLSAuth bool   // Enable TLS Auth for transport security.
	APIKey        string // API key

	DialOptions []grpc.DialOption // Dial options for GRPC.

	parsedAddress *url.URL
}

// Copy a new config, dialOption may shared with old config.
func (c *Config) Copy() Config {
	newConfig := Config{
		Address:       c.Address,
		Username:      c.Username,
		Password:      c.Password,
		DBName:        c.DBName,
		EnableTLSAuth: c.EnableTLSAuth,
	}
	newConfig.DialOptions = make([]grpc.DialOption, 0, len(c.DialOptions))
	newConfig.DialOptions = append(newConfig.DialOptions, c.DialOptions...)
	return newConfig
}

func (c *Config) parse() error {
	// Prepend default fake tcp:// scheme for remote address.
	address := c.Address
	if !regexValidScheme.MatchString(address) {
		address = fmt.Sprintf("tcp://%s", address)
	}

	remoteURL, err := url.Parse(address)
	if err != nil {
		return errors.Wrap(err, "milvus address parse fail")
	}
	// Remote Host should never be empty.
	if remoteURL.Host == "" {
		return errors.New("empty remote host of milvus address")
	}
	// Use DBName in remote url path.
	if c.DBName == "" {
		c.DBName = strings.TrimLeft(remoteURL.Path, "/")
	}
	// Always enable tls auth for https remote url.
	if remoteURL.Scheme == "https" {
		c.EnableTLSAuth = true
	}
	c.parsedAddress = remoteURL
	return nil
}

// Get parsed remote milvus address, should be called after parse was called.
func (c *Config) getParsedAddress() string {
	return c.parsedAddress.Host
}

// syncConfig
type syncConfig struct {
	sync.Mutex
	cfg Config
}

// useDatabase change the inner db name.
func (c *syncConfig) useDatabase(dbName string) {
	c.Lock()
	defer c.Unlock()
	c.cfg.DBName = dbName
}

// Get parsed remote milvus address, should be called after parse was called.
func (c *syncConfig) getParsedAddress() string {
	c.Lock()
	defer c.Unlock()
	return c.cfg.getParsedAddress()
}

// Get parsed grpc dial options, should be called after parse was called.
func (c *syncConfig) getDialOption() []grpc.DialOption {
	c.Lock()
	defer c.Unlock()

	options := c.cfg.DialOptions
	if c.cfg.DialOptions == nil {
		// Add default connection options.
		options = make([]grpc.DialOption, len(DefaultGrpcOpts))
		copy(options, DefaultGrpcOpts)
	}

	// Construct dial option.
	if c.cfg.EnableTLSAuth {
		options = append(options, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})))
	} else {
		options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	options = append(options,
		grpc.WithChainUnaryInterceptor(grpc_retry.UnaryClientInterceptor(
			grpc_retry.WithMax(6),
			grpc_retry.WithBackoff(func(attempt uint) time.Duration {
				return 60 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
			}),
			grpc_retry.WithCodes(codes.Unavailable, codes.ResourceExhausted)),
			RetryOnRateLimitInterceptor(10, func(ctx context.Context, attempt uint) time.Duration {
				return 10 * time.Millisecond * time.Duration(math.Pow(3, float64(attempt)))
			}),
		))

	// Construct username:password field
	if c.cfg.Username != "" && c.cfg.Password != "" {
		options = append(options,
			grpc.WithChainUnaryInterceptor(
				createAuthenticationUnaryInterceptor(c.cfg.Username, c.cfg.Password),
			),
			grpc.WithStreamInterceptor(createAuthenticationStreamInterceptor(c.cfg.Username, c.cfg.Password)),
		)
	}

	// Construct api token
	if c.cfg.APIKey != "" {
		options = append(options, grpc.WithChainUnaryInterceptor(
			createAPIKeyUnaryInteceptor(c.cfg.APIKey),
		))
	}

	// Construct DBName field
	dbNameGetter := func() string {
		c.Lock()
		defer c.Unlock()
		return c.cfg.DBName
	}

	options = append(options,
		grpc.WithChainUnaryInterceptor(
			createDatabaseNameUnaryInterceptor(
				dbNameGetter),
		),
		grpc.WithStreamInterceptor(createDatabaseNameStreamInterceptor(dbNameGetter)),
	)
	return options
}
