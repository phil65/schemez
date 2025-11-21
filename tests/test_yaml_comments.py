"""Tests for YAML with comments functionality."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from schemez.schema import Schema


class DatabaseConfig(Schema):
    """Database configuration."""

    host: str = "localhost"
    """Database host address."""

    port: int = 5432
    """Database port number."""

    username: str
    """Database username for authentication."""

    password: str
    """Database password for authentication."""

    ssl_enabled: bool = False
    """Enable SSL/TLS connection to database."""


class ServerConfig(Schema):
    """Server configuration."""

    name: str
    """Server name identifier."""

    bind_address: str = "0.0.0.0"
    """IP address to bind the server to."""

    port: int = 8080
    """Port number for the server to listen on."""

    debug: bool = False
    """Enable debug mode with verbose logging."""

    database: DatabaseConfig
    """Database connection configuration."""

    allowed_origins: list[str] = Field(default_factory=list)
    """List of allowed CORS origins for web requests."""

    features: dict[str, bool] = Field(default_factory=dict)
    """Feature flags to enable/disable functionality."""


class ServiceType(Schema):
    """Service type configuration."""

    type: Literal["web", "api", "worker"]
    """Type of service to run."""

    replicas: int = 1
    """Number of service replicas to run."""


class ComplexConfig(Schema):
    """Complex nested configuration."""

    version: str = "1.0.0"
    """Configuration version number."""

    server: ServerConfig
    """Main server configuration."""

    services: list[ServiceType] = Field(default_factory=list)
    """List of services to configure."""

    metadata: dict[str, str] = Field(default_factory=dict)
    """Additional metadata for configuration."""


async def test_simple_yaml_with_comments():
    """Test basic YAML with comments for simple fields."""
    db_config = DatabaseConfig(
        host="db.example.com",
        port=3306,
        username="admin",
        password="secret123",
        ssl_enabled=True,
    )

    commented_yaml = db_config.model_dump_yaml(comments=True)

    # Check that all field comments are present
    assert "# Database host address" in commented_yaml
    assert "# Database port number" in commented_yaml
    assert "# Database username for authentication" in commented_yaml
    assert "# Database password for authentication" in commented_yaml
    assert "# Enable SSL/TLS connection to database" in commented_yaml

    # Check that values are preserved
    assert "host: db.example.com" in commented_yaml
    assert "port: 3306" in commented_yaml
    assert "username: admin" in commented_yaml
    assert "password: secret123" in commented_yaml
    assert "ssl_enabled: true" in commented_yaml

    print("=== Simple YAML with Comments ===")
    print(commented_yaml)


async def test_nested_yaml_with_comments():
    """Test YAML with comments for nested objects."""
    server_config = ServerConfig(
        name="main-server",
        bind_address="127.0.0.1",
        port=9000,
        debug=True,
        database=DatabaseConfig(
            host="nested-db.example.com",
            username="nested_user",
            password="nested_pass",
        ),
        allowed_origins=["https://app.com", "https://admin.com"],
        features={"auth": True, "logging": False},
    )

    commented_yaml = server_config.model_dump_yaml(comments=True)

    # Check top-level comments
    assert "# Server name identifier" in commented_yaml
    assert "# IP address to bind the server to" in commented_yaml
    assert "# Port number for the server to listen on" in commented_yaml
    assert "# Enable debug mode with verbose logging" in commented_yaml
    assert "# Database connection configuration" in commented_yaml
    assert "# List of allowed CORS origins for web requests" in commented_yaml
    assert "# Feature flags to enable/disable functionality" in commented_yaml

    # Check nested object comments
    assert "# Database host address" in commented_yaml
    assert "# Database username for authentication" in commented_yaml
    assert "# Database password for authentication" in commented_yaml

    # Check values are preserved
    assert "name: main-server" in commented_yaml
    assert "bind_address: 127.0.0.1" in commented_yaml
    assert "port: 9000" in commented_yaml
    assert "debug: true" in commented_yaml
    assert "host: nested-db.example.com" in commented_yaml
    assert "username: nested_user" in commented_yaml
    assert "- https://app.com" in commented_yaml
    assert "- https://admin.com" in commented_yaml
    assert "auth: true" in commented_yaml
    assert "logging: false" in commented_yaml

    print("\n=== Nested YAML with Comments ===")
    print(commented_yaml)


async def test_complex_nested_yaml_with_comments():
    """Test deeply nested structures with lists and dictionaries."""
    complex_config = ComplexConfig(
        version="2.1.0",
        server=ServerConfig(
            name="production-server",
            database=DatabaseConfig(
                host="prod-db.example.com",
                port=5432,
                username="prod_user",
                password="super_secure",
                ssl_enabled=True,
            ),
            allowed_origins=["https://prod.com"],
            features={"analytics": True, "beta_features": False},
        ),
        services=[
            ServiceType(type="web", replicas=3),
            ServiceType(type="api", replicas=2),
            ServiceType(type="worker", replicas=1),
        ],
        metadata={"environment": "production", "region": "us-east-1"},
    )

    commented_yaml = complex_config.model_dump_yaml(comments=True)

    # Check root level comments
    assert "# Configuration version number" in commented_yaml
    assert "# Main server configuration" in commented_yaml
    assert "# List of services to configure" in commented_yaml
    assert "# Additional metadata for configuration" in commented_yaml

    # Check nested server comments
    assert "# Server name identifier" in commented_yaml
    assert "# Database connection configuration" in commented_yaml

    # Check deeply nested database comments
    assert "# Database host address" in commented_yaml
    assert "# Database port number" in commented_yaml
    assert "# Enable SSL/TLS connection to database" in commented_yaml

    # Array items don't currently get individual field comments
    # This is a known limitation - only the array itself gets a comment
    # Individual object fields within arrays are not yet supported

    # Verify structure is preserved
    assert "version: 2.1.0" in commented_yaml
    assert "name: production-server" in commented_yaml
    assert "host: prod-db.example.com" in commented_yaml
    assert "type: web" in commented_yaml
    assert "replicas: 3" in commented_yaml
    assert "type: api" in commented_yaml
    assert "replicas: 2" in commented_yaml
    assert "environment: production" in commented_yaml
    assert "region: us-east-1" in commented_yaml

    print("\n=== Complex Nested YAML with Comments ===")
    print(commented_yaml)


async def test_empty_collections_with_comments():
    """Test that empty collections still get comments."""
    server_config = ServerConfig(
        name="empty-test",
        database=DatabaseConfig(username="user", password="pass"),
        # allowed_origins and features will be empty
    )

    commented_yaml = server_config.model_dump_yaml(comments=True)

    # Empty collections should still have comments
    assert "# List of allowed CORS origins for web requests" in commented_yaml
    assert "# Feature flags to enable/disable functionality" in commented_yaml

    # Check structure
    assert "allowed_origins: []" in commented_yaml or "allowed_origins:" in commented_yaml
    assert "features: {}" in commented_yaml or "features:" in commented_yaml

    print("\n=== Empty Collections with Comments ===")
    print(commented_yaml)


async def test_multiline_description_truncation():
    """Test that multiline descriptions are truncated to first line."""

    class MultilineDescConfig(Schema):
        """Config with multiline descriptions."""

        field_with_long_desc: str
        """This is a very long description that spans multiple lines.

        It has additional details on the second line.
        And even more information on the third line.
        This should be truncated to just the first line.
        """

        normal_field: int = 42
        """Short description."""

    config = MultilineDescConfig(field_with_long_desc="test")
    commented_yaml = config.model_dump_yaml(comments=True)

    # Should only show first line
    assert "# This is a very long description that spans multiple lines" in commented_yaml
    assert "additional details" not in commented_yaml
    assert "# Short description" in commented_yaml

    print("\n=== Multiline Description Truncation ===")
    print(commented_yaml)


async def test_yaml_without_comments_comparison():
    """Test that regular YAML output doesn't have comments."""
    db_config = DatabaseConfig(host="test.db", username="test_user", password="test_pass")

    regular_yaml = db_config.model_dump_yaml()
    commented_yaml = db_config.model_dump_yaml(comments=True)

    # Regular YAML should not have comments
    assert "#" not in regular_yaml

    # Commented YAML should have comments
    assert "#" in commented_yaml
    assert "# Database host address" in commented_yaml

    print("\n=== Regular YAML (no comments) ===")
    print(regular_yaml)
    print("\n=== Same config with comments ===")
    print(commented_yaml)
