from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import TSVECTOR
from db import Base

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255))
    project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    tsv: Mapped[str] = mapped_column(TSVECTOR, nullable=True)  # populated via trigger
    document = relationship("Document")

class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), unique=True)
    # store as pgvector; alembic migration creates column
    # We'll rely on raw SQL in migration; ORM keeps simple mapping via no attribute for vector
