import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

# Import Visitor only for type checking.
# This avoids unnecessary runtime imports and possible circular imports.
if TYPE_CHECKING:
    from vision.visitor_tracker import Visitor


class VisitRepository:
    """
    Handles all SQLite operations related to completed visits.

    The vision system should not contain raw SQL.
    It only calls save_visit() when a visit finishes.
    """

    def __init__(
        self,
        database_path: str = "data/tass.db",
    ) -> None:
        self.database_path = Path(database_path)

        # Create the data directory automatically if it does not exist.
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # check_same_thread=False will be useful later when FastAPI
        # and the vision processor run in different threads.
        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )

        # Return rows that can be accessed by column name.
        #
        # Example:
        # row["gender"] instead of row[7]
        self.connection.row_factory = sqlite3.Row

        self._configure_database()
        self._create_tables()

    def _configure_database(self) -> None:
        """
        Configure SQLite for safer and more convenient operation.
        """

        cursor = self.connection.cursor()

        # WAL mode allows reading analytics while the vision system
        # is writing new visits. This will be useful for FastAPI later.
        cursor.execute("PRAGMA journal_mode=WAL;")

        # Enable SQLite foreign-key checking.
        cursor.execute("PRAGMA foreign_keys=ON;")

        self.connection.commit()

    def _create_tables(self) -> None:
        """
        Create the visits table when the application starts.

        CREATE TABLE IF NOT EXISTS means existing data is preserved.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,
                tracker_id INTEGER NOT NULL,

                entered_at TEXT NOT NULL,
                left_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,

                age_group TEXT,
                age_confidence REAL,

                gender TEXT,
                gender_confidence REAL,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # This index will make date-based analytics faster.
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_visits_entered_at
            ON visits (entered_at)
            """
        )

        # These indexes will help age and gender charts later.
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_visits_age_group
            ON visits (age_group)
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_visits_gender
            ON visits (gender)
            """
        )

        self.connection.commit()

    def save_visit(
        self,
        session_id: str,
        tracker_id: int,
        visitor: "Visitor",
    ) -> int:
        """
        Save one completed visitor session.

        Returns the permanent database row ID.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO visits (
                session_id,
                tracker_id,
                entered_at,
                left_at,
                duration_seconds,
                age_group,
                age_confidence,
                gender,
                gender_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                tracker_id,

                # ISO 8601 strings are readable and easy to parse later.
                visitor.entered_at.isoformat(),
                visitor.left_at.isoformat(),

                visitor.duration,
                visitor.age_group,
                visitor.age_confidence,
                visitor.gender,
                visitor.gender_confidence,
            ),
        )

        self.connection.commit()

        # SQLite generates this permanent ID automatically.
        return int(cursor.lastrowid)

    def get_visit_count(self) -> int:
        """
        Return the total number of completed visits in the database.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) AS visit_count
            FROM visits
            """
        )

        row = cursor.fetchone()

        if row is None:
            return 0

        return int(row["visit_count"])

    def get_statistics(self) -> dict:
        """
        Return statistics calculated from all completed visits.

        Only completed visits exist in SQLite, so these numbers update
        after a visitor has been missing longer than the configured
        disappearance timeout.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_visits,
                COALESCE(AVG(duration_seconds), 0) AS average_duration,

                SUM(
                    CASE
                        WHEN LOWER(gender) = 'male' THEN 1
                        ELSE 0
                    END
                ) AS male_visits,

                SUM(
                    CASE
                        WHEN LOWER(gender) = 'female' THEN 1
                        ELSE 0
                    END
                ) AS female_visits,

                SUM(
                    CASE
                        WHEN gender IS NULL
                             OR LOWER(gender) NOT IN ('male', 'female')
                        THEN 1
                        ELSE 0
                    END
                ) AS unknown_gender_visits

            FROM visits
            """
        )

        row = cursor.fetchone()

        if row is None:
            return {
                "total_visits": 0,
                "average_duration": 0.0,
                "male_visits": 0,
                "female_visits": 0,
                "unknown_gender_visits": 0,
                "age_groups": {},
            }

        cursor.execute(
            """
            SELECT
                COALESCE(age_group, 'unknown') AS age_group,
                COUNT(*) AS visitor_count
            FROM visits
            GROUP BY COALESCE(age_group, 'unknown')
            ORDER BY age_group
            """
        )

        age_rows = cursor.fetchall()

        age_groups = {
            str(age_row["age_group"]): int(age_row["visitor_count"])
            for age_row in age_rows
        }

        return {
            "total_visits": int(row["total_visits"] or 0),
            "average_duration": float(
                row["average_duration"] or 0.0
            ),
            "male_visits": int(row["male_visits"] or 0),
            "female_visits": int(row["female_visits"] or 0),
            "unknown_gender_visits": int(
                row["unknown_gender_visits"] or 0
            ),
            "age_groups": age_groups,
        }

    def get_recent_visits(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """
        Return the most recently completed visits.

        This method is mainly useful for testing now.
        FastAPI can use a similar method later.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                session_id,
                tracker_id,
                entered_at,
                left_at,
                duration_seconds,
                age_group,
                age_confidence,
                gender,
                gender_confidence
            FROM visits
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def close(self) -> None:
        """
        Close the database connection safely.
        """

        self.connection.close()