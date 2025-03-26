import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import set_tracer_provider
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Azure Monitor imports
try:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter, AzureMonitorLogExporter
    HAS_AZURE_MONITOR = True
    
except ImportError:
    HAS_AZURE_MONITOR = False

def setup_opentelemetry_logging(service_name="L-SARP", connection_string=None):
    """
    Set up OpenTelemetry logging with Azure Monitor integration if available.
    
    Args:
        service_name: The name of the service for resource attribution
        connection_string: The connection string for Azure Monitor
    """
    if not connection_string:
        connection_string = os.getenv("AZURE_APP_INSIGHTS_CONNECTION_STRING")
    
    if not HAS_OPENTELEMETRY:
        logging.warning("OpenTelemetry packages not installed. Skipping OpenTelemetry setup.")
        return

    if not HAS_AZURE_MONITOR:
        logging.warning("Azure Monitor packages not installed. Skipping Azure Monitor setup.")
        return
    
    if not connection_string:
        logging.warning("Azure Monitor connection string not provided. Skipping Azure Monitor setup.")
        return
    
    # Create a resource for service identification
    resource = Resource.create({ResourceAttributes.SERVICE_NAME: service_name})
    
    # Set up tracing
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=connection_string))
    )
    set_tracer_provider(tracer_provider)
    
    # Set up logging
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(AzureMonitorLogExporter(connection_string=connection_string))
    )
    set_logger_provider(logger_provider)
    
    # Create and attach OpenTelemetry logging handler
    handler = LoggingHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)

def setup_logging(config_file: Optional[str] = "configs/logging_conf.ini", 
                 enable_opentelemetry: bool = False,
                 service_name: str = "L-SARP",
                 connection_string: Optional[str] = None) -> tuple[logging.Logger, logging.Logger]:
    """
    Set up logging configuration and return commonly used loggers.
    
    Args:
        config_file: Path to the logging configuration file. Defaults to "configs/logging_conf.ini"
        enable_opentelemetry: Whether to enable OpenTelemetry logging
        service_name: The name of the service for resource attribution
        connection_string: The connection string for Azure Monitor
    
    Returns:
        tuple: (logger_plugins, logger_main) - The two most commonly used loggers
    """
    # Basic logging configuration from file
    logging.config.fileConfig(config_file)
    
    # Set up OpenTelemetry if requested
    if enable_opentelemetry:
        setup_opentelemetry_logging(service_name, connection_string)
    
    # Get commonly used loggers
    logger_plugins = logging.getLogger("plugins")
    logger_main = logging.getLogger("main")
    
    # Set kernel logger to DEBUG level
    logging.getLogger("kernel").setLevel(logging.DEBUG)
    
    return logger_plugins, logger_main

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. If logging hasn't been configured yet, this will configure it
    with the default configuration file.
    
    Args:
        name: Name of the logger to retrieve
    
    Returns:
        logging.Logger: The requested logger
    """
    # Check if logging is configured
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name) 