# FastAPI Basics - Lesson 1

## What is FastAPI?

FastAPI is a modern, fast web framework for building APIs with Python. It's built on top of Starlette (for web parts) and Pydantic (for data validation).

**Why FastAPI?**
- ⚡ Very fast performance (comparable to NodeJS and Go)
- 🎯 Easy to learn and use
- 📝 Automatic interactive API documentation
- 🔒 Built-in data validation
- 🐍 Modern Python features (type hints)

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

- `fastapi` - the framework itself
- `uvicorn` - ASGI server to run your application

## Core Concepts

### 1. The Basics - Creating Your First API

Every FastAPI application starts with creating an app instance and defining routes (endpoints).

**Key Components:**
- **App Instance**: `app = FastAPI()` - creates your application
- **Route/Endpoint**: A URL path like `/` or `/items/5`
- **HTTP Methods**: GET, POST, PUT, DELETE, etc.
- **Path Operation**: The function that runs when a route is accessed

### 2. HTTP Methods (Operations)

- **GET** - Retrieve data (read)
- **POST** - Create new data
- **PUT** - Update existing data (full update)
- **PATCH** - Update existing data (partial update)
- **DELETE** - Remove data

### 3. Request and Response Flow

```
Client → HTTP Request → FastAPI → Your Function → HTTP Response → Client
```

## Example Breakdown

Let's break down a simple endpoint:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

- `@app.get()` - Decorator that tells FastAPI this is a GET endpoint
- `"/items/{item_id}"` - The URL path with a variable
- `async def` - Async function (can also use regular `def`)
- `item_id: int` - Type hint for automatic validation
- `return {...}` - Response data (FastAPI converts to JSON)

## Common Patterns

### Path Parameters
Variables in the URL path: `/users/{user_id}`

### Query Parameters
Parameters after `?` in URL: `/items?skip=0&limit=10`

### Request Body
Data sent in POST/PUT requests (usually JSON)

### Response Model
Define the structure of your API responses

## Next Steps

In the following lessons, we'll cover:
- Lesson 2: Path and Query Parameters
- Lesson 3: Request Body and Data Validation
- Lesson 4: Response Models
- Lesson 5: Database Integration
- Lesson 6: Authentication and Security

---

**Pro Tip:** Always run your FastAPI app with:
```bash
uvicorn main:app --reload
```

The `--reload` flag auto-restarts the server when you change code (use only in development).