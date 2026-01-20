# AGENTS.md - Guidelines for AI Coding Agents

This document provides guidelines for agentic coding assistants working on this repository.

## Build, Lint, and Test Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=papra_llm_manager

# Run a single test file
pytest tests/test_services.py

# Run a single test function
pytest tests/test_services.py::test_create_client

# Run tests matching a pattern
pytest -k "test_create"

# Run tests in verbose mode
pytest -v

# Run async tests (already configured with pytest-asyncio)
pytest tests/test_services.py
```

### Linting and Formatting (Not yet configured)
This project does not currently have linting or formatting configured. Consider adding:
- `ruff` for fast linting and formatting (recommended)
- `black` for code formatting
- `mypy` for type checking

To add ruff:
```bash
pip install ruff
ruff check papra_llm_manager/ tests/
ruff format papra_llm_manager/ tests/
```

## Code Style Guidelines

### Imports
- Group imports: standard library, third-party, local modules
- Use `isort` or organize manually with blank lines between groups
- Use absolute imports for local modules: 
  ```python
  from papra_llm_manager.client import PapraClient
  ```

### Type Hints
- Type hints are **required** for all function signatures
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]`, etc. from `typing` module
- Use `TYPE_CHECKING` for circular imports:
  ```python
  if TYPE_CHECKING:
      from papra_llm_manager.tagger import DocumentTagger
  ```

### Naming Conventions
- Classes: `PascalCase` (e.g., `PapraClient`, `DocumentProcessor`)
- Functions/methods: `snake_case` (e.g., `get_document`, `process_document`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_TAGS`, `DEFAULT_TIMEOUT`)
- Private methods: `_leading_underscore` (e.g., `_should_extract_text`)
- Type variables: `T` or descriptive `T_Something`

### Async/Await
- All API client methods and processor methods are async
- Use `asyncio.gather()` for concurrent operations
- Always use `async` in function definitions that call async methods
- Configure tests with `pytest.mark.asyncio` if needed (already auto-configured)

### Error Handling
- Use custom exception classes from `exceptions.py`
- Catch specific exceptions: `PapraClientError`, `PapraAuthenticationError`, `PapraNotFoundError`
- Use `LLMError` for LLM-related failures
- Return results or raise exceptions - avoid silent failures unless documented
- Log warnings with `print()` for now (consider adding proper logging)

### Pydantic Models
- All data models inherit from `pydantic.BaseModel`
- Use `Field()` for constraints: `Field(ge=0)` for non-negative integers
- Use `@field_validator` for custom validation
- Use `default_factory=list` for mutable default values (avoid `[]` directly)
- Include docstrings for model classes and all fields

### Configuration
- Use `Config.from_env()` to load from environment variables
- Required vars raise `ValueError` if missing
- Optional vars use `os.getenv("KEY", default_value)`
- Convert types: `int(os.getenv("NUM", "10"))`

### Docstrings
- Use Google-style docstrings for classes and methods
- Include `Args:` and `Returns:` sections
- Example:
  ```python
  def upload_document(self, org_id: str, file_path: str) -> Document:
      """Upload a new document to the organization.
      
      Args:
          org_id: The organization ID
          file_path: Path to the file to upload
          
      Returns:
          Document: The created document
      """
  ```

### Service Creation
- Use `PapraServiceFactory` for creating configured services
- Avoid instantiating services directly in business logic
- Pass dependencies explicitly (dependency injection pattern)

### Testing Patterns
- Mock HTTP calls to Papra API with `pytest-mock` or `unittest.mock`
- Mock LLM responses to avoid real API calls
- Test error cases (404, 401, network failures)
- Use fixtures for common test data (see `tests/conftest.py`)
- Test async functions with `async def` test functions

## Architecture Notes

### Core Components
- `config.py`: Environment-based configuration with `.env` support
- `models.py`: Pydantic data models for Papra resources
- `client.py`: Async HTTP client for Papra API
- `llm_handler.py`: LLM abstraction (anthropic, openai, deepseek)
- `tagger.py`: Intelligent tagging system
- `processors.py`: Document processing orchestration
- `cli.py`: Click-based CLI entry point
- `services.py`: Factory for creating configured services

### Design Patterns
- **Factory Pattern**: `PapraServiceFactory` creates configured services
- **Strategy Pattern**: `LLMProvider` abstract base with multiple implementations
- **Repository Pattern**: `PapraClient` encapsulates API access
- **Dependency Injection**: Services receive dependencies via constructor

### Async Patterns
- Use `asyncio.gather(*tasks, return_exceptions=True)` for batch operations
- Process items in configurable batch sizes
- Always close async context managers (use `async with`)
- Use `httpx.AsyncClient` for HTTP requests

### LLM Integration
- Always handle `LLMError` gracefully
- Fallback to empty lists/ dicts on JSON parse failures
- Truncate long text before sending to LLMs (e.g., `text[:4000]`)
- Return only extracted content, no explanations
- Include document name in prompts for context
