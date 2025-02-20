import json
import sqlite3
from datetime import UTC, datetime

from server.models import Conversation, Message


class ConversationStorage:
    """Storage for conversations using SQLite."""

    def __init__(self, db_path: str = "conversations.db"):
        """Initialize the storage with the database path."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    is_user BOOLEAN NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            conn.commit()

    def create_conversation(self, title: str) -> Conversation:
        """Create a new conversation."""
        now = datetime.now(UTC)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?) RETURNING id",
                (title, now, now),
            )
            conversation_id = cursor.fetchone()[0]

            conversation = Conversation(
                id=conversation_id,
                title=title,
                messages=[],
                created_at=now,
                updated_at=now,
            )

        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        """Get a conversation by ID."""
        with sqlite3.connect(self.db_path) as conn:
            # Get conversation
            conv_row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not conv_row:
                return None

            # Get messages
            messages = []
            msg_rows = conn.execute(
                "SELECT text, is_user, timestamp, data FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,),
            ).fetchall()

            for msg_row in msg_rows:
                data = json.loads(msg_row[3]) if msg_row[3] else None
                messages.append(
                    Message(
                        text=msg_row[0],
                        is_user=msg_row[1],
                        timestamp=datetime.fromisoformat(msg_row[2]),
                        data=data,
                    )
                )

            return Conversation(
                id=conv_row[0],
                title=conv_row[1],
                messages=messages,
                created_at=datetime.fromisoformat(conv_row[2]),
                updated_at=datetime.fromisoformat(conv_row[3]),
            )

    def list_conversations(self) -> list[Conversation]:
        """List all conversations."""
        with sqlite3.connect(self.db_path) as conn:
            conversations = []
            conv_rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()

            for conv_row in conv_rows:
                conversations.append(
                    Conversation(
                        id=conv_row[0],
                        title=conv_row[1],
                        messages=[],  # Don't load messages for list view
                        created_at=datetime.fromisoformat(conv_row[2]),
                        updated_at=datetime.fromisoformat(conv_row[3]),
                    )
                )

            return conversations

    def add_message(self, conversation_id: int, message: Message) -> None:
        """Add a message to a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            # Add message
            conn.execute(
                """
                INSERT INTO messages (conversation_id, text, is_user, timestamp, data)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    message.text,
                    message.is_user,
                    message.timestamp,
                    json.dumps(message.data) if message.data else None,
                ),
            )

            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.utcnow(), conversation_id),
            )
            conn.commit()

    def update_conversation_title(self, conversation_id: int, title: str) -> None:
        """Update a conversation's title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.utcnow(), conversation_id),
            )
            conn.commit()

    def clear_messages(self, conversation_id: int) -> None:
        """Clear all messages from a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.utcnow(), conversation_id),
            )
            conn.commit()

    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
