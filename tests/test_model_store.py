"""Tests for ModelStore."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel
import pytest

from schemez.storage import ModelStore


class SimpleModel(BaseModel):
    name: str
    age: int


class ComplexModel(BaseModel):
    name: str
    score: float
    active: bool
    email: str | None = None
    tags: list[str] = []
    metadata: dict[str, int] = {}


class NestedModel(BaseModel):
    title: str
    inner: SimpleModel


class DateTimeModel(BaseModel):
    name: str
    created_at: datetime
    birth_date: date


async def test_insert_and_get():
    async with ModelStore(SimpleModel) as store:
        user = SimpleModel(name="Alice", age=30)
        row_id = await store.insert(user)
        assert row_id == 1

        retrieved = await store.get(row_id)
        assert retrieved is not None
        assert retrieved.name == "Alice"
        assert retrieved.age == 30  # noqa: PLR2004


async def test_get_nonexistent():
    async with ModelStore(SimpleModel) as store:
        result = await store.get(999)
        assert result is None


async def test_insert_many():
    async with ModelStore(SimpleModel) as store:
        users = [
            SimpleModel(name="Alice", age=30),
            SimpleModel(name="Bob", age=25),
            SimpleModel(name="Charlie", age=35),
        ]
        ids = await store.insert_many(users)
        assert ids == [1, 2, 3]

        all_users = await store.all()
        assert len(all_users) == 3  # noqa: PLR2004


async def test_delete():
    async with ModelStore(SimpleModel) as store:
        user = SimpleModel(name="Alice", age=30)
        row_id = await store.insert(user)

        deleted = await store.delete(row_id)
        assert deleted is True

        retrieved = await store.get(row_id)
        assert retrieved is None


async def test_delete_nonexistent():
    async with ModelStore(SimpleModel) as store:
        deleted = await store.delete(999)
        assert deleted is False


async def test_count():
    async with ModelStore(SimpleModel) as store:
        assert await store.count() == 0

        await store.insert(SimpleModel(name="Alice", age=30))
        assert await store.count() == 1

        await store.insert(SimpleModel(name="Bob", age=25))
        assert await store.count() == 2  # noqa: PLR2004


async def test_query():
    async with ModelStore(SimpleModel) as store:
        await store.insert_many([
            SimpleModel(name="Alice", age=30),
            SimpleModel(name="Bob", age=30),
            SimpleModel(name="Charlie", age=25),
        ])

        results = await store.query(age=30)
        assert len(results) == 2  # noqa: PLR2004
        assert all(u.age == 30 for u in results)  # noqa: PLR2004

        results = await store.query(name="Alice", age=30)
        assert len(results) == 1
        assert results[0].name == "Alice"


async def test_query_empty_filters():
    async with ModelStore(SimpleModel) as store:
        await store.insert(SimpleModel(name="Alice", age=30))
        results = await store.query()
        assert len(results) == 1


async def test_query_unknown_field():
    async with ModelStore(SimpleModel) as store:
        with pytest.raises(ValueError, match="Unknown field"):
            await store.query(nonexistent=42)


async def test_complex_types():
    async with ModelStore(ComplexModel) as store:
        model = ComplexModel(
            name="Test",
            score=3.14,
            active=True,
            email="test@example.com",
            tags=["a", "b", "c"],
            metadata={"x": 1, "y": 2},
        )
        row_id = await store.insert(model)

        retrieved = await store.get(row_id)
        assert retrieved is not None
        assert retrieved.name == "Test"
        assert retrieved.score == pytest.approx(3.14)
        assert retrieved.active is True
        assert retrieved.email == "test@example.com"
        assert retrieved.tags == ["a", "b", "c"]
        assert retrieved.metadata == {"x": 1, "y": 2}


async def test_optional_none():
    async with ModelStore(ComplexModel) as store:
        model = ComplexModel(name="Test", score=1.0, active=False)
        row_id = await store.insert(model)

        retrieved = await store.get(row_id)
        assert retrieved is not None
        assert retrieved.email is None


async def test_nested_model():
    async with ModelStore(NestedModel) as store:
        model = NestedModel(
            title="Outer",
            inner=SimpleModel(name="Inner", age=10),
        )
        row_id = await store.insert(model)

        retrieved = await store.get(row_id)
        assert retrieved is not None
        assert retrieved.title == "Outer"
        assert isinstance(retrieved.inner, SimpleModel)
        assert retrieved.inner.name == "Inner"
        assert retrieved.inner.age == 10  # noqa: PLR2004


async def test_datetime_handling():
    async with ModelStore(DateTimeModel) as store:
        now = datetime(2024, 1, 15, 10, 30, 0)
        today = date(2024, 1, 15)
        model = DateTimeModel(name="Test", created_at=now, birth_date=today)

        row_id = await store.insert(model)
        retrieved = await store.get(row_id)

        assert retrieved is not None
        assert retrieved.created_at == now
        assert retrieved.birth_date == today


async def test_custom_table_name():
    async with ModelStore(SimpleModel, table_name="custom_users") as store:
        assert store._table_name == "custom_users"
        await store.insert(SimpleModel(name="Alice", age=30))
        assert await store.count() == 1


async def test_context_manager_required():
    store = ModelStore(SimpleModel)
    with pytest.raises(RuntimeError, match="must be used as async context manager"):
        await store.insert(SimpleModel(name="Alice", age=30))


async def test_all_empty():
    async with ModelStore(SimpleModel) as store:
        results = await store.all()
        assert results == []


async def test_schema_storage_and_reconstruction(tmp_path):
    """Test that schema is stored and model can be reconstructed."""
    db_path = tmp_path / "test.db"

    # Create and populate database
    async with ModelStore(SimpleModel, db_path) as store:
        await store.insert(SimpleModel(name="Alice", age=30))
        await store.insert(SimpleModel(name="Bob", age=25))

    # Reopen without providing model type
    async with await ModelStore.open(db_path) as store:
        # Model type should be reconstructed
        assert store.model_type is not None
        assert store.model_type.__name__ == "SimpleModel"

        # Should be able to query
        all_items = await store.all()
        assert len(all_items) == 2  # noqa: PLR2004
        assert all_items[0].name == "Alice"
        assert all_items[0].age == 30  # noqa: PLR2004


async def test_schema_reconstruction_complex_model(tmp_path):
    """Test reconstruction of model with complex types."""
    db_path = tmp_path / "complex.db"

    async with ModelStore(ComplexModel, db_path) as store:
        await store.insert(
            ComplexModel(
                name="Test",
                score=3.14,
                active=True,
                tags=["a", "b"],
                metadata={"x": 1},
            )
        )

    async with await ModelStore.open(db_path) as store:
        items = await store.all()
        assert len(items) == 1
        assert items[0].name == "Test"
        assert items[0].score == pytest.approx(3.14)
        assert items[0].active is True
        assert items[0].tags == ["a", "b"]
        assert items[0].metadata == {"x": 1}


async def test_open_nonexistent_database():
    """Test that opening nonexistent database raises error."""
    with pytest.raises(FileNotFoundError, match="Database not found"):
        await ModelStore.open("/nonexistent/path.db")


async def test_model_type_required_for_memory():
    """Test that model_type is required for in-memory databases."""
    with pytest.raises(ValueError, match="model_type is required for in-memory"):
        ModelStore(model_type=None, path=":memory:")


async def test_model_type_required_for_new_file(tmp_path):
    """Test that model_type is required when creating new database."""
    db_path = tmp_path / "new.db"
    with pytest.raises(ValueError, match="model_type is required when database doesn't exist"):
        ModelStore(model_type=None, path=db_path)


async def test_json_schema_property():
    """Test that json_schema property returns the model's schema."""
    async with ModelStore(SimpleModel) as store:
        schema = store.json_schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]


async def test_model_type_property_before_enter():
    """Test that model_type works before entering context when provided."""
    store = ModelStore(SimpleModel)
    # Before entering, model_type should still work since it was provided
    assert store.model_type == SimpleModel
