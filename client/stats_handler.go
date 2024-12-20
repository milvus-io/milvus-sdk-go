package client

import (
	"context"
	"sync"
	"sync/atomic"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdk "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"google.golang.org/grpc/stats"
)

var (
	dynamicClientHandler *dynamicOtelGrpcStatsHandler
	initClientOnce       sync.Once
)

// dynamicOtelGrpcStatsHandler wraps otelgprc.StatsHandler
// to implement runtime configuration update.
type dynamicOtelGrpcStatsHandler struct {
	handler atomic.Pointer[stats.Handler]
}

func getDynamicClientStatsHandler() *dynamicOtelGrpcStatsHandler {
	initClientOnce.Do(func() {
		// init trace noop provider
		tp := sdk.NewTracerProvider(
			sdk.WithBatcher(nil),
			sdk.WithResource(resource.NewWithAttributes(
				semconv.SchemaURL,
				semconv.ServiceNameKey.String("Client"),
			)),
			sdk.WithSampler(sdk.ParentBased(
				sdk.TraceIDRatioBased(1),
			)),
		)
		otel.SetTracerProvider(tp)
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

		// init trace stats handler
		statsHandler := otelgrpc.NewClientHandler(
			otelgrpc.WithTracerProvider(otel.GetTracerProvider()),
		)

		dynamicClientHandler = &dynamicOtelGrpcStatsHandler{}
		dynamicClientHandler.handler.Store(&statsHandler)
	})

	return dynamicClientHandler
}

func (h *dynamicOtelGrpcStatsHandler) getHandler() stats.Handler {
	return *h.handler.Load()
}

func (h *dynamicOtelGrpcStatsHandler) setHandler(handler stats.Handler) {
	h.handler.Store(&handler)
}

// TagRPC can attach some information to the given context.
// The context used for the rest lifetime of the RPC will be derived from
// the returned context.
func (h *dynamicOtelGrpcStatsHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	handler := h.getHandler()
	if handler == nil {
		return ctx
	}

	return handler.TagRPC(ctx, info)
}

// HandleRPC processes the RPC stats.
func (h *dynamicOtelGrpcStatsHandler) HandleRPC(ctx context.Context, stats stats.RPCStats) {
	handler := h.getHandler()
	if handler == nil {
		return
	}

	handler.HandleRPC(ctx, stats)
}

// TagConn can attach some information to the given context.
// The returned context will be used for stats handling.
// For conn stats handling, the context used in HandleConn for this
// connection will be derived from the context returned.
// For RPC stats handling,
//   - On server side, the context used in HandleRPC for all RPCs on this
//
// connection will be derived from the context returned.
//   - On client side, the context is not derived from the context returned.
func (h *dynamicOtelGrpcStatsHandler) TagConn(ctx context.Context, tagInfo *stats.ConnTagInfo) context.Context {
	handler := h.getHandler()
	if handler == nil {
		return ctx
	}

	return handler.TagConn(ctx, tagInfo)
}

// HandleConn processes the Conn stats.
func (h *dynamicOtelGrpcStatsHandler) HandleConn(ctx context.Context, stats stats.ConnStats) {
	handler := h.getHandler()
	if handler == nil {
		return
	}

	handler.HandleConn(ctx, stats)
}
