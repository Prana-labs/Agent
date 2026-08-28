import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, DateTime, Integer,
    ForeignKey, Enum as SAEnum, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database import Base


def new_uuid():
    return str(uuid.uuid4())


# =========================================================
# USERS
# =========================================================

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=True, index=True)

    # Phase 2 — Auth (nullable until then)
    hashed_password = Column(String(255), nullable=True)
    auth_provider = Column(String(50), nullable=True, default="local")  # local | google | github

    # Phase 3 — Payments (nullable until then)
    stripe_customer_id = Column(String(255), nullable=True, unique=True)
    plan = Column(String(50), nullable=True, default="free")  # free | pro | enterprise

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")


# =========================================================
# SESSIONS  (one per PDF upload)
# =========================================================

class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid, index=True)
    thread_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    filenames = Column(JSON, nullable=False, default=list)  # list of uploaded PDF names

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")
    uploaded_files = relationship("UploadedFile", back_populates="session", cascade="all, delete-orphan")


# =========================================================
# MESSAGES  (chat history per session)
# =========================================================

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid, index=True)
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(SAEnum("human", "ai", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="messages")


# =========================================================
# UPLOADED FILES  (log of each PDF per session)
# =========================================================

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid, index=True)
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="uploaded_files")


# =========================================================
# SUBSCRIPTIONS  (Phase 3 — Stripe)
# =========================================================

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid, index=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stripe_subscription_id = Column(String(255), unique=True, nullable=True)
    status = Column(String(50), nullable=False, default="active")  # active | cancelled | past_due
    plan = Column(String(50), nullable=False, default="free")
    current_period_end = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="subscriptions")
