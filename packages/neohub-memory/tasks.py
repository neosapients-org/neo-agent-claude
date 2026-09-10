"""
Invoke tasks for Neo Memory Hub development workflow.

Usage:
    uv run invoke --list                  # List all available tasks
    uv run invoke dev-setup              # Complete development setup
    uv run invoke infra-up               # Start infrastructure services
    uv run invoke infra-down             # Stop infrastructure services
    uv run invoke test                   # Run all tests
    uv run invoke lint                   # Run linting
    uv run invoke format                 # Format code
"""

import os
import sys
import time
from pathlib import Path

from invoke import task


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCKER_COMPOSE = PROJECT_ROOT / "docker" / "docker-compose.dev.yml"


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

@task
def sync(c):
    """
    Sync dependencies using uv.
    
    This ensures all dependencies are installed and up to date.
    """
    print("📦 Syncing dependencies with uv...")
    c.run("uv sync --all-extras", pty=True)
    print("✅ Dependencies synced successfully!")


@task
def env_setup(c):
    """
    Setup environment file if it doesn't exist.
    
    Copies .env.example to .env if .env doesn't exist.
    """
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    
    if env_file.exists():
        print("✅ .env file already exists")
    else:
        print("📝 Creating .env file from .env.example...")
        env_example.read_text().replace(
            "sk-your-openai-api-key-here",
            "YOUR_OPENAI_API_KEY"
        )
        c.run(f"cp {env_example} {env_file}")
        print("⚠️  Please update .env file with your actual API keys!")
        print(f"   Edit: {env_file}")


# =============================================================================
# INFRASTRUCTURE MANAGEMENT
# =============================================================================

@task
def infra_up(c, profile=None):
    """
    Start infrastructure services (Milvus, Redis, PostgreSQL).
    
    Args:
        profile: Optional docker-compose profile (e.g., 'ui' for admin UIs)
    
    Examples:
        uv run invoke infra-up              # Start core services
        uv run invoke infra-up --profile=ui # Start with admin UIs
    """
    print("🚀 Starting infrastructure services...")
    
    cmd = f"docker-compose -f {DOCKER_COMPOSE} up -d"
    if profile:
        cmd += f" --profile {profile}"
    
    c.run(cmd, pty=True)
    
    print("\n⏳ Waiting for services to be ready...")
    time.sleep(5)
    
    print("\n📊 Service Status:")
    c.run(f"docker-compose -f {DOCKER_COMPOSE} ps", pty=True)
    
    print("\n✅ Infrastructure is ready!")
    print("\n📍 Service URLs:")
    print("   - Milvus:     localhost:19530")
    print("   - Redis:      localhost:6379")
    print("   - PostgreSQL: localhost:5432")
    print("   - MinIO:      http://localhost:9001 (admin: minioadmin/minioadmin)")
    if profile == "ui":
        print("   - Attu (Milvus UI): http://localhost:8001")


@task
def infra_down(c):
    """
    Stop infrastructure services.
    """
    print("🛑 Stopping infrastructure services...")
    c.run(f"docker-compose -f {DOCKER_COMPOSE} down", pty=True)
    print("✅ Infrastructure stopped!")


@task
def infra_restart(c):
    """
    Restart infrastructure services.
    """
    print("🔄 Restarting infrastructure services...")
    infra_down(c)
    time.sleep(2)
    infra_up(c)


@task
def infra_logs(c, service=None, follow=False):
    """
    View infrastructure logs.
    
    Args:
        service: Specific service name (e.g., 'milvus-standalone', 'redis')
        follow: Follow log output (tail -f style)
    
    Examples:
        uv run invoke infra-logs                     # All logs
        uv run invoke infra-logs --service=redis     # Redis logs only
        uv run invoke infra-logs --follow            # Follow all logs
    """
    cmd = f"docker-compose -f {DOCKER_COMPOSE} logs"
    if follow:
        cmd += " -f"
    if service:
        cmd += f" {service}"
    
    c.run(cmd, pty=True)


@task
def infra_clean(c, volumes=False):
    """
    Clean infrastructure (stop and remove containers).
    
    Args:
        volumes: Also remove volumes (WARNING: deletes all data!)
    
    Examples:
        uv run invoke infra-clean               # Remove containers
        uv run invoke infra-clean --volumes     # Remove containers and data
    """
    print("🧹 Cleaning infrastructure...")
    cmd = f"docker-compose -f {DOCKER_COMPOSE} down"
    if volumes:
        print("⚠️  WARNING: This will delete all data!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return
        cmd += " -v"
    
    c.run(cmd, pty=True)
    print("✅ Cleanup complete!")


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

@task
def db_migrate(c):
    """
    Run database migrations.
    """
    print("🔄 Running database migrations...")
    c.run("uv run alembic upgrade head", pty=True)
    print("✅ Migrations complete!")


@task
def db_seed(c):
    """
    Seed database with sample data for testing.
    """
    print("🌱 Seeding database with sample data...")
    
    seed_script = """
import asyncio
from neo_memory_hub import MemoryHub
from neo_memory_hub.dto import StoreRequest

async def seed():
    async with MemoryHub() as hub:
        # Sample memories for testing
        samples = [
            StoreRequest(
                content="User prefers dark mode for the interface",
                tenant_id="demo_tenant",
                user_id="user_001",
                metadata={"category": "preferences", "priority": "high"}
            ),
            StoreRequest(
                content="User's favorite programming language is Python",
                tenant_id="demo_tenant",
                user_id="user_001",
                metadata={"category": "skills", "priority": "medium"}
            ),
            StoreRequest(
                content="User is working on a memory management project",
                tenant_id="demo_tenant",
                user_id="user_001",
                metadata={"category": "projects", "priority": "high"}
            ),
        ]
        
        for req in samples:
            result = await hub.store(req)
            print(f"✓ Stored: {req.content[:50]}...")
        
        print(f"\\n✅ Seeded {len(samples)} sample memories!")

if __name__ == "__main__":
    asyncio.run(seed())
"""
    
    # Create temporary seed script
    seed_file = PROJECT_ROOT / "seed_temp.py"
    seed_file.write_text(seed_script)
    
    try:
        c.run(f"uv run python {seed_file}", pty=True)
    finally:
        seed_file.unlink()


# =============================================================================
# TESTING
# =============================================================================

@task
def test(c, coverage=True, verbose=True, marker=None):
    """
    Run tests.
    
    Args:
        coverage: Run with coverage report
        verbose: Verbose output
        marker: Run specific test marker (unit, integration, e2e)
    
    Examples:
        uv run invoke test                    # All tests with coverage
        uv run invoke test --marker=unit      # Unit tests only
        uv run invoke test --no-coverage      # Without coverage
    """
    print("🧪 Running tests...")
    
    cmd = "uv run pytest"
    
    if verbose:
        cmd += " -v"
    
    if marker:
        cmd += f" -m {marker}"
    
    if coverage:
        cmd += " --cov --cov-report=term-missing --cov-report=html"
    
    c.run(cmd, pty=True)
    
    if coverage:
        print("\n📊 Coverage report generated: coverage_html/index.html")


@task
def test_unit(c):
    """
    Run unit tests only (fast, no external dependencies).
    """
    print("🧪 Running unit tests...")
    c.run("uv run pytest -v -m unit", pty=True)


@task
def test_integration(c):
    """
    Run integration tests (requires infrastructure).
    """
    print("🧪 Running integration tests...")
    print("⚠️  Make sure infrastructure is running (invoke infra-up)")
    c.run("uv run pytest -v -m integration", pty=True)


@task
def test_watch(c):
    """
    Run tests in watch mode (re-run on file changes).
    """
    print("🧪 Running tests in watch mode...")
    c.run("uv run pytest-watch", pty=True)


# =============================================================================
# CODE QUALITY
# =============================================================================

@task
def format(c, check=False):
    """
    Format code with Black and Ruff.
    
    Args:
        check: Only check, don't modify files
    
    Examples:
        uv run invoke format           # Format all code
        uv run invoke format --check   # Check formatting only
    """
    print("🎨 Formatting code...")
    
    black_cmd = "uv run black"
    ruff_cmd = "uv run ruff check"
    
    if check:
        black_cmd += " --check"
    else:
        ruff_cmd += " --fix"
    
    print("Running Black...")
    c.run(f"{black_cmd} {SRC_DIR} {TESTS_DIR}", pty=True)
    
    print("\nRunning Ruff...")
    c.run(f"{ruff_cmd} {SRC_DIR} {TESTS_DIR}", pty=True)
    
    print("✅ Formatting complete!")


@task
def lint(c):
    """
    Run all linters (Ruff, Black check, MyPy).
    """
    print("🔍 Running linters...")
    
    print("\n1️⃣ Ruff...")
    c.run(f"uv run ruff check {SRC_DIR} {TESTS_DIR}", pty=True, warn=True)
    
    print("\n2️⃣ Black (check only)...")
    c.run(f"uv run black --check {SRC_DIR} {TESTS_DIR}", pty=True, warn=True)
    
    print("\n3️⃣ MyPy (type checking)...")
    c.run(f"uv run mypy {SRC_DIR}", pty=True, warn=True)
    
    print("\n✅ Linting complete!")


@task
def type_check(c):
    """
    Run type checking with MyPy.
    """
    print("🔍 Running type checker...")
    c.run(f"uv run mypy {SRC_DIR}", pty=True)


# =============================================================================
# DEVELOPMENT WORKFLOW
# =============================================================================

@task(pre=[sync, env_setup])
def dev_setup(c, with_ui=False, seed_data=False):
    """
    Complete development environment setup.
    
    This will:
    1. Sync all dependencies
    2. Setup .env file
    3. Start infrastructure services
    4. Run database migrations
    5. Optionally seed sample data
    
    Args:
        with_ui: Start with admin UIs (Attu for Milvus)
        seed_data: Seed database with sample data
    
    Examples:
        uv run invoke dev-setup                        # Basic setup
        uv run invoke dev-setup --with-ui              # With UIs
        uv run invoke dev-setup --with-ui --seed-data  # Full setup
    """
    print("\n" + "="*60)
    print("🚀 NEO MEMORY HUB - Development Setup")
    print("="*60 + "\n")
    
    # Start infrastructure
    print("\n📦 Step 1: Starting Infrastructure...")
    infra_up(c, profile="ui" if with_ui else None)
    
    # Wait for services
    print("\n⏳ Waiting for services to be fully ready...")
    time.sleep(10)
    
    # Run migrations (if needed)
    print("\n📦 Step 2: Database Migrations...")
    # db_migrate(c)  # Uncomment when alembic is set up
    
    # Seed data
    if seed_data:
        print("\n📦 Step 3: Seeding Sample Data...")
        db_seed(c)
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("   1. Update your .env file with API keys")
    print("   2. Run tests: uv run invoke test")
    print("   3. Start coding! 🚀")
    print("\n📍 Useful Commands:")
    print("   - uv run invoke infra-logs         # View logs")
    print("   - uv run invoke test               # Run tests")
    print("   - uv run invoke lint               # Check code quality")
    print("   - uv run invoke --list             # See all commands")
    print()


@task
def clean(c, all_cache=False):
    """
    Clean build artifacts and cache.
    
    Args:
        all_cache: Also clean uv cache and pytest cache
    
    Examples:
        uv run invoke clean              # Clean build artifacts
        uv run invoke clean --all-cache  # Clean everything
    """
    print("🧹 Cleaning build artifacts...")
    
    # Python cache
    c.run("find . -type d -name '__pycache__' -exec rm -rf {} +", warn=True)
    c.run("find . -type f -name '*.pyc' -delete", warn=True)
    c.run("find . -type f -name '*.pyo' -delete", warn=True)
    c.run("find . -type f -name '*.egg-info' -exec rm -rf {} +", warn=True)
    
    # Build artifacts
    c.run("rm -rf build dist .eggs", warn=True)
    
    # Coverage
    c.run("rm -rf .coverage coverage_html htmlcov", warn=True)
    
    if all_cache:
        print("🧹 Cleaning all caches...")
        c.run("rm -rf .pytest_cache .mypy_cache .ruff_cache", warn=True)
    
    print("✅ Cleanup complete!")


@task
def docs_serve(c, port=8000):
    """
    Serve documentation locally.
    
    Args:
        port: Port to serve on (default: 8000)
    """
    print(f"📚 Serving documentation on http://localhost:{port}")
    c.run(f"python -m http.server {port} -d docs", pty=True)


# =============================================================================
# RELEASE & BUILD
# =============================================================================

@task
def build(c):
    """
    Build distribution packages.
    """
    print("📦 Building distribution packages...")
    clean(c)
    c.run("uv build", pty=True)
    print("✅ Build complete! Check dist/ directory")


@task(pre=[lint, test])
def pre_commit(c):
    """
    Run all pre-commit checks (lint + test).
    
    Use this before committing code.
    """
    print("\n✅ All pre-commit checks passed!")
    print("Ready to commit! 🚀")


# =============================================================================
# DEMO & TESTING SCRIPTS
# =============================================================================

@task
def demo_gradio(c):
    """
    Launch the interactive Gradio demo with LangGraph agent.
    
    This demo shows:
    - Real LangGraph agent with create_react_agent
    - Live memory storage tracking (what's stored in each memory type)
    - Live memory retrieval tracking (which memories are retrieved)
    - Context visualization (how memories personalize responses)
    
    Opens at: http://localhost:7860
    """
    print("🚀 Launching Gradio demo...")
    print("   Demo will open at: http://localhost:7860")
    print("   Press Ctrl+C to stop\n")
    c.run("uv run python scripts/gradio_demo.py", pty=True)


@task
def test_e2e(c, cleanup=True):
    """
    Run end-to-end tests with live data.
    
    Args:
        cleanup: Clean databases before testing (default: True)
    
    Examples:
        uv run invoke test-e2e                # With cleanup
        uv run invoke test-e2e --no-cleanup   # Skip cleanup
    """
    print("🧪 Running E2E tests...")
    
    if cleanup:
        print("🧹 Cleaning databases first...")
        c.run("uv run python scripts/cleanup_databases.py --force", pty=True)
        print()
    
    print("▶️  Running test suite...")
    c.run("uv run python scripts/run_e2e_tests.py", pty=True)


@task
def test_langgraph(c):
    """
    Run LangGraph agent test script.
    
    Tests:
    - Universal Connector
    - LangGraph Adapter
    - LangChain Adapter
    - create_react_agent
    - Memory persistence
    """
    print("🧪 Testing LangGraph integration...")
    c.run("uv run python scripts/test_langgraph_agent.py", pty=True)


@task
def cleanup_db(c, force=False):
    """
    Clean all databases (Milvus, Redis, PostgreSQL).
    
    Args:
        force: Skip confirmation prompt
    
    Examples:
        uv run invoke cleanup-db         # With confirmation
        uv run invoke cleanup-db --force # No confirmation
    """
    print("🧹 Cleaning databases...")
    
    cmd = "uv run python scripts/cleanup_databases.py"
    if force:
        cmd += " --force"
    
    c.run(cmd, pty=True)


# =============================================================================
# UTILITIES
# =============================================================================

@task
def shell(c):
    """
    Start an interactive Python shell with the project context.
    """
    print("🐍 Starting interactive shell...")
    c.run("uv run python", pty=True)


@task
def check_env(c):
    """
    Check environment configuration and dependencies.
    """
    print("🔍 Checking environment...\n")
    
    env_file = PROJECT_ROOT / ".env"
    
    print("📋 Environment File:")
    if env_file.exists():
        print(f"   ✅ .env exists at {env_file}")
    else:
        print(f"   ❌ .env not found! Run: uv run invoke env-setup")
    
    print("\n📦 Python & UV:")
    c.run("python --version", pty=True)
    c.run("uv --version", pty=True)
    
    print("\n🐳 Docker:")
    c.run("docker --version", pty=True, warn=True)
    c.run("docker-compose --version", pty=True, warn=True)
    
    print("\n📊 Infrastructure Status:")
    c.run(f"docker-compose -f {DOCKER_COMPOSE} ps", pty=True, warn=True)


@task
def info(c):
    """
    Display project information and status.
    """
    print("\n" + "="*60)
    print("📊 NEO MEMORY HUB - Project Information")
    print("="*60 + "\n")
    
    print("📁 Project Structure:")
    print(f"   Root:   {PROJECT_ROOT}")
    print(f"   Source: {SRC_DIR}")
    print(f"   Tests:  {TESTS_DIR}")
    
    print("\n🔧 Quick Start:")
    print("   uv run invoke dev-setup      # Complete setup")
    print("   uv run invoke test           # Run tests")
    print("   uv run invoke lint           # Check code")
    
    print("\n📚 Documentation:")
    print("   uv run invoke --list         # All commands")
    print("   uv run invoke <command> --help  # Command help")
    
    print()
