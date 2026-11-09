"""followme tenant run admission

Adds the tenant isolation column to threads and a client-stable operation
key plus frozen intent digest to runs, with one partial unique index that
makes ``(tenant, owner, thread, operation key)`` a single logical run.

Revision ID: f7c2d9b1a4e6
Revises: a3f7c1d9e2b4
Create Date: 2026-11-09 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "f7c2d9b1a4e6"
down_revision = "a3f7c1d9e2b4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable, NO server default, NO backfill: a pre-existing row has no
    # provable tenant, and defaulting it would silently grant 'default'
    # ownership of another tenant's data. New writes always set a trusted,
    # non-empty tenant; legacy NULL rows are readable only through the
    # explicit single-tenant compatibility branch (see tenant_scope()).
    op.add_column("thread", sa.Column("tenant_id", sa.Text(), nullable=True))
    op.create_index("idx_thread_tenant_user", "thread", ["tenant_id", "user_id"], unique=False)

    op.add_column("runs", sa.Column("tenant_id", sa.Text(), nullable=True))
    op.add_column("runs", sa.Column("operation_key", sa.Text(), nullable=True))
    op.add_column("runs", sa.Column("intent_digest", sa.Text(), nullable=True))
    op.add_column("runs", sa.Column("admission_version", sa.Integer(), nullable=True))

    # Partial: legacy rows (and runs created without a client key before this
    # migration) keep operation_key NULL and are never deduplicated, so an
    # empty key cannot collide across runs in one thread.
    op.create_index(
        "uq_runs_tenant_owner_thread_opkey",
        "runs",
        ["tenant_id", "user_id", "thread_id", "operation_key"],
        unique=True,
        postgresql_where=sa.text("operation_key IS NOT NULL"),
    )
    op.create_index("idx_runs_tenant_user", "runs", ["tenant_id", "user_id"], unique=False)

    # Crons: tenant attribution for new rows only. Deliberately nullable and
    # NOT backfilled: legacy cron rows have no provable tenant, and defaulting
    # them would silently attribute other tenants' schedules to 'default'.
    op.add_column("crons", sa.Column("tenant_id", sa.Text(), nullable=True))
    op.create_index("idx_cron_tenant_user", "crons", ["tenant_id", "user_id"], unique=False)

    # Assistants: tenant attribution for new rows only, same no-backfill rule.
    # System assistants (user_id='system') stay tenant-independent and are
    # matched by their own proven sharing rule, not by this column.
    op.add_column("assistant", sa.Column("tenant_id", sa.Text(), nullable=True))
    op.create_index("idx_assistant_tenant_user", "assistant", ["tenant_id", "user_id"], unique=False)

    # Rebuild assistant uniqueness per (tenant, owner). Two partial indexes keep
    # the guarantee for pre-attribution rows (tenant NULL), which a single index
    # containing tenant_id would lose because Postgres treats NULLs as distinct.
    # The config expression (md5(config::text)) is preserved verbatim.
    op.drop_index("idx_assistant_user_assistant", table_name="assistant")
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")
    op.create_index(
        "idx_assistant_tenant_user_assistant",
        "assistant",
        ["tenant_id", "user_id", "assistant_id"],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NOT NULL"),
    )
    op.create_index(
        "idx_assistant_user_assistant_legacy",
        "assistant",
        ["user_id", "assistant_id"],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NULL"),
    )
    op.create_index(
        "idx_assistant_tenant_user_graph_config",
        "assistant",
        ["tenant_id", "user_id", "graph_id", sa.text("md5(config::text)")],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NOT NULL"),
    )
    op.create_index(
        "idx_assistant_user_graph_config_legacy",
        "assistant",
        ["user_id", "graph_id", sa.text("md5(config::text)")],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NULL"),
    )


def downgrade() -> None:
    # Recreate the original owner-wide uniqueness FIRST: if several tenants
    # legitimately hold the same (owner, graph, config), this fails and the
    # transaction aborts. That is the safe outcome — the downgrade never deletes
    # or merges tenant rows to make the old index fit.
    op.create_index(
        "idx_assistant_user_assistant",
        "assistant",
        ["user_id", "assistant_id"],
        unique=True,
    )
    op.create_index(
        "idx_assistant_user_graph_config",
        "assistant",
        ["user_id", "graph_id", sa.text("md5(config::text)")],
        unique=True,
    )
    op.drop_index("idx_assistant_user_graph_config_legacy", table_name="assistant")
    op.drop_index("idx_assistant_tenant_user_graph_config", table_name="assistant")
    op.drop_index("idx_assistant_user_assistant_legacy", table_name="assistant")
    op.drop_index("idx_assistant_tenant_user_assistant", table_name="assistant")
    op.drop_index("idx_assistant_tenant_user", table_name="assistant")
    op.drop_column("assistant", "tenant_id")
    op.drop_index("idx_cron_tenant_user", table_name="crons")
    op.drop_column("crons", "tenant_id")
    op.drop_index("idx_runs_tenant_user", table_name="runs")
    op.drop_index("uq_runs_tenant_owner_thread_opkey", table_name="runs")
    op.drop_column("runs", "admission_version")
    op.drop_column("runs", "intent_digest")
    op.drop_column("runs", "operation_key")
    op.drop_column("runs", "tenant_id")
    op.drop_index("idx_thread_tenant_user", table_name="thread")
    op.drop_column("thread", "tenant_id")
