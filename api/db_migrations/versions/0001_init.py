from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # documents
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=True),
    )

    # chunks
    op.create_table(
        'chunks',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('document_id', sa.Integer(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('tsv', sa.dialects.postgresql.TSVECTOR(), nullable=True),
    )
    # index for tsvector and trigram for fallback
    op.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv)""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops)""")

    # trigger to populate tsvector
    op.execute("""
    CREATE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
    begin
      new.tsv := to_tsvector('english', coalesce(new.content,''));
      return new;
    end
    $$ LANGUAGE plpgsql;
    """)
    op.execute("""
    CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE
    ON chunks FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();
    """)

    # embeddings with pgvector(768) (nomic-embed-text default size is 768)
    op.create_table(
        'embeddings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('chunk_id', sa.Integer(), sa.ForeignKey('chunks.id', ondelete='CASCADE'), nullable=False, unique=True),
    )
    op.execute("""ALTER TABLE embeddings ADD COLUMN vector vector(768)""")
    op.execute("""CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_l2_ops) WITH (lists = 100)""")

def downgrade():
    op.drop_table('embeddings')
    op.execute("DROP TRIGGER IF EXISTS chunks_tsv_update ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_tsv_update")
    op.drop_table('chunks')
    op.drop_table('documents')
    # leave extensions in place
