"""Async SQLite storage for Pydantic models."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, get_args, get_origin

from pydantic import BaseModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    import aiosqlite


# SQLite type mapping for Python types
TYPE_MAP: dict[type, str] = {
    str: "TEXT",
    int: "INTEGER",
    float: "REAL",
    bool: "INTEGER",
    bytes: "BLOB",
    datetime: "TEXT",
    date: "TEXT",
}

META_TABLE = "_model_meta"


def _get_sqlite_type(annotation: Any) -> tuple[str, bool]:
    """Get SQLite type and nullability from a Python type annotation.

    Returns:
        Tuple of (sqlite_type, is_nullable)
    """
    origin = get_origin(annotation)

    # Handle Optional (Union with None) / X | None
    if origin is type(None):
        return "TEXT", True

    args = get_args(annotation)

    # Check for Optional pattern: Union[X, None] or X | None
    if args and type(None) in args:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            inner_type, _ = _get_sqlite_type(non_none_args[0])
            return inner_type, True
        # Multiple non-None types -> JSON
        return "TEXT", True

    # Direct type lookup
    if annotation in TYPE_MAP:
        return TYPE_MAP[annotation], False

    # Complex types (list, dict, nested BaseModel, etc.) -> JSON
    return "TEXT", False


def _is_complex_type(annotation: Any) -> bool:
    """Check if a type needs JSON serialization."""
    args = get_args(annotation)

    # Handle Optional
    if args and type(None) in args:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _is_complex_type(non_none_args[0])
        return True

    return annotation not in TYPE_MAP


def _serialize_value(value: Any, annotation: Any) -> Any:  # noqa: PLR0911
    """Serialize a Python value for SQLite storage."""
    if value is None:
        return None

    if _is_complex_type(annotation):
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        return json.dumps(value)

    match value:
        case datetime():
            return value.isoformat()
        case date():
            return value.isoformat()
        case bool():
            return int(value)
        case bytes():
            return value
        case _:
            return value


def _deserialize_value(value: Any, annotation: Any) -> Any:  # noqa: PLR0911
    """Deserialize a SQLite value to Python."""
    if value is None:
        return None

    args = get_args(annotation)

    # Handle Optional
    if args and type(None) in args:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _deserialize_value(value, non_none_args[0])

    # Handle datetime/date stored as TEXT
    if annotation is datetime:
        return datetime.fromisoformat(value)
    if annotation is date:
        return date.fromisoformat(value)

    # Handle bool stored as INTEGER
    if annotation is bool:
        return bool(value)

    # Handle complex types stored as JSON
    if _is_complex_type(annotation):
        data = json.loads(value) if isinstance(value, str) else value
        # Check if it's a BaseModel subclass
        actual_type = annotation
        if args and type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                actual_type = non_none[0]
        if isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
            return actual_type.model_validate(data)
        return data

    return value


class ModelStore[T: BaseModel]:
    """Type-safe async SQLite storage for Pydantic models.

    Can be initialized with a model type, or opened from an existing database
    that has the schema stored.

    Example:
        ```python
        class User(BaseModel):
            name: str
            age: int
            email: str | None = None

        # Create new database with known type
        async with ModelStore(User, "users.db") as store:
            user_id = await store.insert(User(name="Alice", age=30))
            user = await store.get(user_id)
            all_users = await store.all()

        # Open existing database, reconstruct type from stored schema
        async with await ModelStore.open("users.db") as store:
            users = await store.all()  # Returns reconstructed model instances
        ```
    """

    def __init__(
        self,
        model_type: type[T] | None = None,
        path: str | Path = ":memory:",
        table_name: str | None = None,
    ) -> None:
        """Initialize the model store.

        Args:
            model_type: The Pydantic model class to store (required for new DBs)
            path: SQLite database path (":memory:" for in-memory)
            table_name: Custom table name (defaults to lowercase model name)

        Raises:
            ValueError: If model_type is None and path doesn't exist or is :memory:
        """
        self._path = str(path)
        self._connection: aiosqlite.Connection | None = None
        self._model_type: type[T] | None = model_type
        self._table_name = table_name
        self._columns: list[tuple[str, str, bool, Any]] | None = None
        self._import_path: str | None = None

        # Validate: can't create new DB without model_type
        if model_type is None:
            if self._path == ":memory:":
                msg = "model_type is required for in-memory databases"
                raise ValueError(msg)
            if not Path(self._path).exists():
                msg = f"model_type is required when database doesn't exist: {self._path}"
                raise ValueError(msg)
        else:
            self._table_name = table_name or model_type.__name__.lower()
            self._columns = self._build_column_definitions()
            module = model_type.__module__
            if module != "__main__":
                self._import_path = f"{module}.{model_type.__name__}"

    @classmethod
    async def open(cls, path: str | Path) -> ModelStore[BaseModel]:
        """Open an existing database and reconstruct the model from stored schema.

        Args:
            path: Path to existing SQLite database

        Returns:
            ModelStore with reconstructed model type

        Raises:
            FileNotFoundError: If database doesn't exist
            ValueError: If database has no stored schema
        """
        path = Path(path)
        if not path.exists():
            msg = f"Database not found: {path}"
            raise FileNotFoundError(msg)

        return cls(model_type=None, path=path)  # type: ignore[return-value]

    @property
    def model_type(self) -> type[T]:
        """The model class (original or reconstructed)."""
        if self._model_type is None:
            msg = "Model type not yet available. Use async context manager first."
            raise RuntimeError(msg)
        return self._model_type

    @property
    def json_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the stored model."""
        return self.model_type.model_json_schema()

    @property
    def import_path(self) -> str | None:
        """Get the import path for the stored model, if available."""
        return self._import_path

    def _build_column_definitions(self) -> list[tuple[str, str, bool, Any]]:
        """Build column definitions from model fields.

        Returns:
            List of (name, sqlite_type, is_nullable, annotation) tuples
        """
        if self._model_type is None:
            msg = "Cannot build columns without model_type"
            raise RuntimeError(msg)

        columns: list[tuple[str, str, bool, Any]] = []
        for name, field_info in self._model_type.model_fields.items():
            annotation = field_info.annotation
            sqlite_type, is_nullable = _get_sqlite_type(annotation)
            # Also nullable if field has a default
            if field_info.default is not None or field_info.default_factory is not None:
                is_nullable = True
            columns.append((name, sqlite_type, is_nullable, annotation))
        return columns

    def _create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement."""
        if self._columns is None:
            msg = "Columns not initialized"
            raise RuntimeError(msg)

        column_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for name, sqlite_type, is_nullable, _ in self._columns:
            null_clause = "" if is_nullable else " NOT NULL"
            column_defs.append(f"{name} {sqlite_type}{null_clause}")
        columns_sql = ", ".join(column_defs)
        return f"CREATE TABLE IF NOT EXISTS {self._table_name} ({columns_sql})"

    def _create_meta_table_sql(self) -> str:
        """Generate CREATE TABLE SQL for metadata."""
        return f"""
            CREATE TABLE IF NOT EXISTS {META_TABLE} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """

    async def _store_schema(self, conn: aiosqlite.Connection) -> None:
        """Store the model's JSON schema in the metadata table."""
        if self._model_type is None:
            return

        schema = self._model_type.model_json_schema()
        model_name = self._model_type.__name__
        module = self._model_type.__module__
        import_path = f"{module}.{model_name}" if module != "__main__" else None

        await conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES (?, ?)",
            ("schema", json.dumps(schema)),
        )
        await conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES (?, ?)",
            ("model_name", model_name),
        )
        await conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES (?, ?)",
            ("table_name", self._table_name),
        )
        if import_path:
            await conn.execute(
                f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES (?, ?)",
                ("import_path", import_path),
            )

    async def _load_schema(self, conn: aiosqlite.Connection) -> None:
        """Load and reconstruct model from stored schema."""
        from schemez.schema_to_type import json_schema_to_pydantic_class

        # Check if meta table exists
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (META_TABLE,),
        )
        if await cursor.fetchone() is None:
            msg = "Database has no metadata table. Cannot reconstruct model."
            raise ValueError(msg)

        # Load schema and model name
        cursor = await conn.execute(
            f"SELECT key, value FROM {META_TABLE} WHERE key IN (?, ?, ?, ?)",
            ("schema", "model_name", "table_name", "import_path"),
        )
        rows = await cursor.fetchall()
        meta = {row[0]: row[1] for row in rows}

        if "schema" not in meta:
            msg = "No schema found in database metadata"
            raise ValueError(msg)

        schema = json.loads(meta["schema"])
        model_name = meta.get("model_name", "DynamicModel")
        self._table_name = meta.get("table_name", model_name.lower())
        self._import_path = meta.get("import_path")

        # Reconstruct the model class
        self._model_type = json_schema_to_pydantic_class(schema, class_name=model_name)  # type: ignore[assignment]
        self._columns = self._build_column_definitions()

    async def __aenter__(self) -> Self:
        """Open connection and create/load table."""
        import aiosqlite

        self._connection = await aiosqlite.connect(self._path)
        self._connection.row_factory = aiosqlite.Row

        if self._model_type is None:
            # Opening existing DB - reconstruct model from schema
            await self._load_schema(self._connection)
        else:
            # Creating new or opening with known type
            await self._connection.execute(self._create_meta_table_sql())
            await self._connection.execute(self._create_table_sql())
            await self._store_schema(self._connection)
            await self._connection.commit()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    def _ensure_connected(self) -> aiosqlite.Connection:
        """Ensure we have an active connection."""
        if self._connection is None:
            msg = "ModelStore must be used as async context manager"
            raise RuntimeError(msg)
        return self._connection

    def _instance_to_row(self, instance: T) -> dict[str, Any]:
        """Convert model instance to row dict with serialized values."""
        if self._columns is None:
            msg = "Columns not initialized"
            raise RuntimeError(msg)

        row = {}
        for name, _, _, annotation in self._columns:
            value = getattr(instance, name)
            row[name] = _serialize_value(value, annotation)
        return row

    def _row_to_instance(self, row: aiosqlite.Row) -> T:
        """Convert database row to model instance."""
        if self._columns is None or self._model_type is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        data = {}
        for name, _, _, annotation in self._columns:
            value = row[name]
            data[name] = _deserialize_value(value, annotation)
        return self._model_type.model_validate(data)

    async def insert(self, instance: T) -> int:
        """Insert a model instance.

        Args:
            instance: The model instance to insert

        Returns:
            The inserted row's id
        """
        conn = self._ensure_connected()
        row = self._instance_to_row(instance)
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO {self._table_name} ({columns}) VALUES ({placeholders})"
        cursor = await conn.execute(sql, list(row.values()))
        await conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def insert_many(self, instances: Sequence[T]) -> list[int]:
        """Insert multiple model instances.

        Args:
            instances: The model instances to insert

        Returns:
            List of inserted row ids
        """
        ids = []
        for instance in instances:
            row_id = await self.insert(instance)
            ids.append(row_id)
        return ids

    async def get(self, row_id: int) -> T | None:
        """Fetch a model instance by id.

        Args:
            row_id: The row id to fetch

        Returns:
            The model instance, or None if not found
        """
        conn = self._ensure_connected()
        sql = f"SELECT * FROM {self._table_name} WHERE id = ?"
        cursor = await conn.execute(sql, (row_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_instance(row)

    async def all(self) -> list[T]:
        """Fetch all model instances.

        Returns:
            List of all model instances
        """
        conn = self._ensure_connected()
        sql = f"SELECT * FROM {self._table_name}"
        cursor = await conn.execute(sql)
        rows = await cursor.fetchall()
        return [self._row_to_instance(row) for row in rows]

    async def delete(self, row_id: int) -> bool:
        """Delete a model instance by id.

        Args:
            row_id: The row id to delete

        Returns:
            True if a row was deleted, False otherwise
        """
        conn = self._ensure_connected()
        sql = f"DELETE FROM {self._table_name} WHERE id = ?"
        cursor = await conn.execute(sql, (row_id,))
        await conn.commit()
        return cursor.rowcount > 0

    async def count(self) -> int:
        """Count total rows in the table.

        Returns:
            Number of rows
        """
        conn = self._ensure_connected()
        sql = f"SELECT COUNT(*) FROM {self._table_name}"
        cursor = await conn.execute(sql)
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def query(self, **filters: Any) -> list[T]:
        """Query instances with simple equality filters.

        Args:
            **filters: Field-value pairs for WHERE clause (equality only)

        Returns:
            List of matching model instances

        Example:
            ```python
            users = await store.query(age=30, name="Alice")
            ```
        """
        conn = self._ensure_connected()

        if not filters:
            return await self.all()

        if self._columns is None:
            msg = "Columns not initialized"
            raise RuntimeError(msg)

        # Build WHERE clause
        conditions = []
        values = []
        annotations = {name: ann for name, _, _, ann in self._columns}

        for field, value in filters.items():
            if field not in annotations:
                msg = f"Unknown field: {field}"
                raise ValueError(msg)
            conditions.append(f"{field} = ?")
            values.append(_serialize_value(value, annotations[field]))

        where_clause = " AND ".join(conditions)
        sql = f"SELECT * FROM {self._table_name} WHERE {where_clause}"
        cursor = await conn.execute(sql, values)
        rows = await cursor.fetchall()
        return [self._row_to_instance(row) for row in rows]
