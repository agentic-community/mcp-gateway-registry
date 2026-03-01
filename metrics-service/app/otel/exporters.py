from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR
from prometheus_client import REGISTRY as PROM_REGISTRY
import logging
from ..config import settings

logger = logging.getLogger(__name__)

# Unregister default Prometheus collectors that emit process/platform/gc
# metrics with timestamps. These cause "out of order samples" rejections
# in AMP when the container restarts (new timestamps < stored timestamps).
for collector in [PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR]:
    try:
        PROM_REGISTRY.unregister(collector)
    except Exception:
        pass


def setup_otel():
    """Setup OpenTelemetry metric providers and exporters."""
    readers = []

    try:
        # Create resource with service name
        resource = Resource.create(attributes={SERVICE_NAME: settings.OTEL_SERVICE_NAME})

        # Create a View for duration histograms with configurable boundaries
        duration_view = View(
            instrument_name="*_duration_seconds",
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=settings.HISTOGRAM_BUCKET_BOUNDARIES,
            ),
        )

        # Setup Prometheus exporter if enabled
        if settings.OTEL_PROMETHEUS_ENABLED:
            # Start Prometheus HTTP server
            start_http_server(
                port=settings.OTEL_PROMETHEUS_PORT, addr=settings.METRICS_SERVICE_HOST
            )  # nosec B104

            # Create PrometheusMetricReader (no endpoint parameter needed)
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
            logger.info(
                f"Prometheus metrics exporter enabled on port {settings.OTEL_PROMETHEUS_PORT}"
            )

        # Setup OTLP exporter if endpoint configured
        if settings.OTEL_OTLP_ENDPOINT:
            otlp_exporter = OTLPMetricExporter(endpoint=f"{settings.OTEL_OTLP_ENDPOINT}/v1/metrics")
            otlp_reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=30000,  # 30 seconds
            )
            readers.append(otlp_reader)
            logger.info(f"OTLP metrics exporter enabled for {settings.OTEL_OTLP_ENDPOINT}")

        # Create MeterProvider with configured readers and resource
        if readers:
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=readers,
                views=[duration_view],
            )
            metrics.set_meter_provider(meter_provider)
            logger.info("OpenTelemetry metrics configured successfully")
        else:
            logger.warning("No OpenTelemetry exporters configured")

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}")
        # Don't fail startup, just log the error
        pass
