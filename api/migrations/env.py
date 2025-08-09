import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your Base from the app, and load DB URL from settings
from db import Base
from settings import get_settings

# Alembic Config object
config = context.config

fileConfig(config.config_file_name)

# Load database URL from settings and inject into Alembic config
settings = get_settings()
url = getattr(settings, 'database_url', None) or getattr(settings, 'DATABASE_URL', None)
if not url:
    raise RuntimeError("Database URL not found in settings (expected 'database_url' or 'DATABASE_URL').")

# Ensure the URL is available to engine_from_config
config.set_main_option("sqlalchemy.url", str(url))

# Set the metadata for autogenerate
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    context.configure(
        url=str(url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()