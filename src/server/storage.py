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
                    context TEXT,
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
                "INSERT INTO conversations (title, context, created_at, updated_at) VALUES (?, ?, ?, ?) RETURNING id",
                (title, None, now, now),
            )
            conversation_id = cursor.fetchone()[0]

            conversation = Conversation(
                id=conversation_id,
                title=title,
                messages=[],
                context=None,
                created_at=now,
                updated_at=now,
            )

        return conversation

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        """Get a conversation by ID."""
        with sqlite3.connect(self.db_path) as conn:
            # Get conversation
            conv_row = conn.execute(
                "SELECT id, title, context, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not conv_row:
                return None

            # Get messages
            messages = []
            msg_rows = conn.execute(
                "SELECT id, text, is_user, timestamp, data FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,),
            ).fetchall()

            for msg_row in msg_rows:
                data = json.loads(msg_row[4]) if msg_row[4] else None
                messages.append(
                    Message(
                        id=msg_row[0],
                        text=msg_row[1],
                        is_user=msg_row[2],
                        timestamp=datetime.fromisoformat(msg_row[3]),
                        data=data,
                        conversation_id=conversation_id,
                    )
                )

            return Conversation(
                id=conv_row[0],
                title=conv_row[1],
                context=conv_row[2],
                messages=messages,
                created_at=datetime.fromisoformat(conv_row[3]),
                updated_at=datetime.fromisoformat(conv_row[4]),
            )

    def list_conversations(self) -> list[Conversation]:
        """List all conversations."""
        with sqlite3.connect(self.db_path) as conn:
            conversations = []
            conv_rows = conn.execute(
                "SELECT id, title, context, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()

            for conv_row in conv_rows:
                conversations.append(
                    Conversation(
                        id=conv_row[0],
                        title=conv_row[1],
                        context=conv_row[2],
                        messages=[],  # Don't load messages for list view
                        created_at=datetime.fromisoformat(conv_row[3]),
                        updated_at=datetime.fromisoformat(conv_row[4]),
                    )
                )

            return conversations

    def add_message(self, conversation_id: int, message: Message) -> int:
        """Add a message to a conversation. Returns the ID of the created message."""
        with sqlite3.connect(self.db_path) as conn:
            # Add message
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, text, is_user, timestamp, data)
                VALUES (?, ?, ?, ?, ?) RETURNING id
                """,
                (
                    conversation_id,
                    message.text,
                    message.is_user,
                    message.timestamp,
                    json.dumps(message.data) if message.data else None,
                ),
            )
            message_id = cursor.fetchone()[0]

            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.utcnow(), conversation_id),
            )
            conn.commit()

            return message_id

    def update_conversation_title(self, conversation_id: int, title: str) -> None:
        """Update a conversation's title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.utcnow(), conversation_id),
            )
            conn.commit()

    def update_conversation_context(
        self, conversation_id: int, context: str | None
    ) -> None:
        """Update a conversation's context."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET context = ?, updated_at = ? WHERE id = ?",
                (context, datetime.utcnow(), conversation_id),
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

    def get_message(self, message_id: int) -> Message | None:
        """Get a message by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT conversation_id, text, is_user, timestamp, data
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()

            if not row:
                return None

            data = json.loads(row[4]) if row[4] else None
            return Message(
                id=message_id,
                text=row[1],
                is_user=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                data=data,
                conversation_id=row[0],
            )

    def update_message(self, message_id: int, message: Message) -> None:
        """Update a message's data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE messages 
                SET text = ?, is_user = ?, timestamp = ?, data = ?
                WHERE id = ?
                """,
                (
                    message.text,
                    message.is_user,
                    message.timestamp,
                    json.dumps(message.data) if message.data else None,
                    message_id,
                ),
            )

            # Update the conversation's updated_at timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (datetime.utcnow(), message.conversation_id),
            )
            conn.commit()
