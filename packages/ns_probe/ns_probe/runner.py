"""
Runtime runner for ns_probe (Client-Side Agent).
This module provides a CLI entry point to run any python script with:
1. OTel Auto-Instrumentation (FastAPI, HTTPX, etc.)
2. LLM Instrumentation (OpenAI)
3. Agent Monkey-Patching (ReActAgent)
4. OTLP Export (Sends data to Collector)
"""

import sys
import os
import logging
import runpy
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ns_probe.runner")


def setup_opentelemetry():
    """Configure OpenTelemetry SDK and OTLP Exporter."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        # Default to localhost collector if not set
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", "ns-agent-service"),
                "service.version": "0.1.0",
            }
        )

        provider = TracerProvider(resource=resource)

        # Use OTLP Exporter instead of local DB
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)

        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        logger.info(f"OpenTelemetry configured. Sending traces to: {endpoint}")
        return trace.get_tracer("ns_probe.runner")

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}", exc_info=True)
        return None


def instrument_libraries():
    """Auto-instrument standard libraries."""
    try:
        # LLM
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
            logger.info("Instrumented OpenAI")
        except ImportError:
            pass

        # Web & HTTP — skip, handled by instrumentors.py with URL filtering.
        # The OTel contrib HTTPXClientInstrumentor/RequestsInstrumentor do not
        # support URL exclusion, so we rely on ns_probe's own wrappers instead.
        pass

        # DB
        try:
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument()
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            RedisInstrumentor().instrument()
            logger.info("Instrumented SQLAlchemy/Redis")
        except ImportError:
            pass

        # System
        try:
            # from opentelemetry.instrumentation.logging import LoggingInstrumentor
            # LoggingInstrumentor().instrument()
            from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

            SystemMetricsInstrumentor().instrument()
            logger.info("Instrumented System Metrics (Logging disabled to prevent recursion)")
        except ImportError:
            pass

    except Exception as e:
        logger.warning(f"Partial instrumentation failure: {e}")


import importlib.abc
import importlib.util


class AgentLoader(importlib.abc.Loader):
    """
    Custom Loader that delegates to the real loader and then patches the module.
    """

    def __init__(self, real_loader, tracer):
        self.real_loader = real_loader
        self.tracer = tracer

    def create_module(self, spec):
        # Delegate to real loader if it supports create_module, else None (default behavior)
        if hasattr(self.real_loader, "create_module"):
            return self.real_loader.create_module(spec)
        return None

    def exec_module(self, module):
        # 1. Execute the module using the real loader
        if hasattr(self.real_loader, "exec_module"):
            self.real_loader.exec_module(module)

        # 2. Apply the Patch to BaseAgent
        try:
            if hasattr(module, "BaseAgent"):
                self._patch_base_agent(module.BaseAgent)
                logger.info("Successfully patched BaseAgent via Import Hook")
            else:
                logger.warning(f"BaseAgent not found in {module.__name__}")
        except Exception as e:
            logger.error(f"Failed to patch BaseAgent: {e}")

    def _patch_base_agent(self, BaseAgentClass):
        """Wrap the execute method of BaseAgent."""
        original_execute = BaseAgentClass.execute
        tracer = self.tracer

        async def traced_execute(self, task: Any, **kwargs):
            # Imported here, not at module scope: ns-probe-run is an optional
            # entry point and the OTel SDK is not a hard dependency, so importing
            # it on `import ns_probe` would make the SDK fail closed for everyone
            # who never uses the CLI. The handler below used to reach for a
            # binding made inside setup_opentelemetry() — a different function —
            # so it raised NameError over the top of whatever the agent raised.
            from opentelemetry import trace

            # Use agent name or ID for span name
            agent_name = getattr(self.config, "name", "UnknownAgent")
            span_name = f"Agent Step: {agent_name}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.id", getattr(self.config, "agent_id", "unknown"))
                span.set_attribute("agent.task", str(task))

                try:
                    result = await original_execute(self, task, **kwargs)
                    span.set_attribute("agent.output", str(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise

        BaseAgentClass.execute = traced_execute


class AgentImportHook(importlib.abc.MetaPathFinder):
    """
    Finder that intercepts the import of ns_core.agents.base.
    """

    def __init__(self, tracer):
        self.tracer = tracer

    def find_spec(self, fullname, path, target=None):
        if fullname != "ns_core.agents.base":
            return None

        # Temporarily remove self to find the real spec
        if self in sys.meta_path:
            sys.meta_path.remove(self)

        try:
            # Find the real spec (e.g. from file system)
            real_spec = importlib.util.find_spec(fullname)
        finally:
            # Always add self back
            sys.meta_path.insert(0, self)

        if not real_spec or not real_spec.loader:
            return None

        # Create a new spec that uses our custom AgentLoader
        # We wrap the real loader so we can execute it and then patch
        loader = AgentLoader(real_spec.loader, self.tracer)

        # We must copy the spec to avoid modifying the cached real spec if it exists
        # But importlib.util.spec_from_loader creates a new one
        return importlib.util.spec_from_loader(fullname, loader)


def install_import_hook(tracer):
    """Install the import hook into sys.meta_path."""
    if not tracer:
        return

    # Check if already installed
    for finder in sys.meta_path:
        if isinstance(finder, AgentImportHook):
            return

    hook = AgentImportHook(tracer)
    sys.meta_path.insert(0, hook)
    logger.info("Installed Agent Import Hook")


def main():
    if len(sys.argv) < 2:
        print("Usage: ns-probe-run <script_path> [args...]")
        sys.exit(1)

    script_path = sys.argv[1]

    # Adjust sys.argv so the target script sees its own arguments
    sys.argv = sys.argv[1:]

    # 1. Setup OTel
    tracer = setup_opentelemetry()

    # 2. Instrument Libraries
    instrument_libraries()

    # 3. Install Import Hook (Robust Patching)
    install_import_hook(tracer)

    # 4. Run Target Script
    logger.info(f"Starting target script: {script_path}")

    try:
        # Ensure the script's directory is in sys.path
        script_dir = os.path.dirname(os.path.abspath(script_path))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
