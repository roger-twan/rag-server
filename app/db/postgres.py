from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings


def _database_url() -> str:
    if settings.DATABASE_URL.startswith("postgresql://"):
        return settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    return settings.DATABASE_URL


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),)

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_preview: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class RetrievalTraceRecord(Base):
    __tablename__ = "retrieval_traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    pinecone_score: Mapped[float | None] = mapped_column(nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


engine = create_engine(_database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def upsert_document(
    *,
    doc_id: str,
    source: str,
    path: str,
    title: str,
    content_hash: str,
    meta: dict[str, Any],
    session: Session,
) -> None:
    record = session.get(DocumentRecord, doc_id)
    if record is None:
        record = DocumentRecord(
            id=doc_id,
            source=source,
            path=path,
            title=title,
            content_hash=content_hash,
            meta=meta,
        )
        session.add(record)
        return

    record.source = source
    record.path = path
    record.title = title
    record.content_hash = content_hash
    record.meta = meta
    record.updated_at = datetime.now(timezone.utc)


def replace_document_chunks(
    *,
    doc_id: str,
    chunks: list[dict[str, Any]],
    session: Session,
) -> None:
    session.query(ChunkRecord).filter(ChunkRecord.document_id == doc_id).delete()
    for chunk in chunks:
        session.add(
            ChunkRecord(
                id=chunk["id"],
                document_id=doc_id,
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                text_preview=chunk["text_preview"],
                content_hash=chunk["content_hash"],
                meta=chunk.get("meta", {}),
            )
        )


def delete_document(doc_id: str) -> None:
    with session_scope() as session:
        session.query(ChunkRecord).filter(ChunkRecord.document_id == doc_id).delete()
        record = session.get(DocumentRecord, doc_id)
        if record is not None:
            session.delete(record)


def document_has_chunks(doc_id: str) -> bool:
    with session_scope() as session:
        return (
            session.query(ChunkRecord.id)
            .filter(ChunkRecord.document_id == doc_id)
            .limit(1)
            .one_or_none()
            is not None
        )


def get_chunks_by_ids(chunk_ids: list[str]) -> dict[str, ChunkRecord]:
    if not chunk_ids:
        return {}
    with session_scope() as session:
        rows = session.scalars(select(ChunkRecord).where(ChunkRecord.id.in_(chunk_ids))).all()
        return {row.id: row for row in rows}


def get_neighbor_chunks(doc_id: str, chunk_index: int, window: int = 1) -> list[ChunkRecord]:
    start = max(0, chunk_index - window)
    end = chunk_index + window
    with session_scope() as session:
        return list(
            session.scalars(
                select(ChunkRecord)
                .where(
                    ChunkRecord.document_id == doc_id,
                    ChunkRecord.chunk_index >= start,
                    ChunkRecord.chunk_index <= end,
                )
                .order_by(ChunkRecord.chunk_index)
            ).all()
        )


def ensure_conversation(conversation_id: str | None = None) -> str:
    with session_scope() as session:
        if conversation_id:
            record = session.get(ConversationRecord, conversation_id)
            if record is not None:
                return record.id

        record = ConversationRecord(id=conversation_id or str(uuid4()))
        session.add(record)
        session.flush()
        return record.id


def add_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    rewritten_query: str | None = None,
) -> str:
    with session_scope() as session:
        record = MessageRecord(
            conversation_id=conversation_id,
            role=role,
            content=content,
            rewritten_query=rewritten_query,
        )
        session.add(record)
        session.flush()
        return record.id


def get_recent_messages(conversation_id: str, limit: int = 8) -> list[MessageRecord]:
    with session_scope() as session:
        rows = session.scalars(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.created_at.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))


def add_retrieval_traces(
    *,
    message_id: str,
    query: str,
    chunks: list[dict[str, Any]],
) -> None:
    with session_scope() as session:
        for rank, chunk in enumerate(chunks, start=1):
            session.add(
                RetrievalTraceRecord(
                    message_id=message_id,
                    query=query,
                    chunk_id=chunk["chunk_id"],
                    pinecone_score=chunk.get("score"),
                    rerank_score=chunk.get("rerank_score"),
                    rank=rank,
                )
            )
