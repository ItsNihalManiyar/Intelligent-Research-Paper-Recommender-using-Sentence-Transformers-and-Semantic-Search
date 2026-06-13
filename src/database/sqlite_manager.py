import os
import sqlite3
from typing import List, Dict, Any

class SQLiteManager:
    def __init__(self, db_path: str = "database/favorites.db"):
        self.db_path = db_path
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.initialize_db()
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Returns a sqlite3 connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self) -> None:
        """
        Initializes the database and creates the favorites table if it doesn't exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT UNIQUE,
                    title TEXT NOT NULL,
                    year INTEGER
                )
            """)
            conn.commit()

    def add_favorite(self, paper_id: str, title: str, year: int) -> bool:
        """
        Adds a paper to favorites. Returns True if successful, False if already exists.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO favorites (paper_id, title, year) VALUES (?, ?, ?)",
                    (paper_id, title, year)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            # Paper already in favorites due to UNIQUE constraint
            return False
        except Exception as e:
            print(f"Error adding favorite: {e}")
            return False

    def remove_favorite(self, paper_id: str) -> bool:
        """
        Removes a paper from favorites.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM favorites WHERE paper_id = ?",
                    (paper_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing favorite: {e}")
            return False

    def get_favorites(self) -> List[Dict[str, Any]]:
        """
        Retrieves all favorite papers.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, paper_id, title, year FROM favorites ORDER BY id DESC")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error fetching favorites: {e}")
            return []

if __name__ == "__main__":
    # Test database
    manager = SQLiteManager("database/test_favorites.db")
    print("Database initialized.")
    added = manager.add_favorite("123.456", "Test Crop Yield Paper", 2023)
    print(f"Add status: {added}")
    favs = manager.get_favorites()
    print("Favorites list:", favs)
    removed = manager.remove_favorite("123.456")
    print(f"Remove status: {removed}")
    # Cleanup test db
    if os.path.exists("database/test_favorites.db"):
        os.remove("database/test_favorites.db")
