# FastAPI Lesson 2 - Request Body & Data Validation

## What You'll Learn

In this lesson, we'll cover:
- Pydantic models for data validation
- Request body handling
- Response models
- Data validation rules
- Error handling

## What is Pydantic?

Pydantic is a data validation library that uses Python type hints. FastAPI uses Pydantic to:
- ✅ Validate incoming data automatically
- 📝 Generate API documentation automatically
- 🔄 Convert data types automatically
- ❌ Return clear error messages when data is invalid

## Request Body Basics

When clients send data to your API (POST, PUT, PATCH), they send it in the **request body**, usually as JSON.

```json
{
  "name": "Laptop",
  "price": 999.99,
  "is_available": true
}
```

## Pydantic Models (Schemas)

Think of Pydantic models as blueprints for your data. They define:
- What fields are required
- What type each field should be
- Validation rules for each field
- Default values

### Basic Model Structure

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_available: bool = True  # Default value
```

## Field Validation

Pydantic provides powerful validation through the `Field` class:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    price: float = Field(..., gt=0, le=10000)
    quantity: int = Field(default=0, ge=0)
```

**Field Parameters:**
- `...` (Ellipsis) = Required field (no default)
- `default=value` = Optional with default value
- `min_length`, `max_length` = String length validation
- `gt`, `ge`, `lt`, `le` = Number comparisons (greater than, less than, etc.)
- `regex` = Pattern matching for strings

## Response Models

Response models define what your API returns. Benefits:
- 🔒 Filter sensitive data (passwords, internal IDs)
- 📊 Guarantee consistent response structure
- 📝 Improve API documentation

```python
@app.post("/items/", response_model=ItemResponse)
async def create_item(item: Item):
    # Your logic here
    return item
```

## Nested Models

You can nest Pydantic models inside each other:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    email: str
    address: Address  # Nested model
```

## Common Validation Patterns

### 1. Email Validation
```python
from pydantic import EmailStr

class User(BaseModel):
    email: EmailStr  # Validates email format
```

### 2. Enums (Fixed Choices)
```python
from enum import Enum

class Category(str, Enum):
    electronics = "electronics"
    furniture = "furniture"
    clothing = "clothing"

class Product(BaseModel):
    category: Category  # Must be one of the enum values
```

### 3. Optional Fields
```python
from typing import Optional

class Item(BaseModel):
    name: str
    description: Optional[str] = None  # Can be null/missing
```

## HTTP Status Codes

Use proper status codes in responses:
- **200** - OK (successful GET, PUT, PATCH)
- **201** - Created (successful POST)
- **204** - No Content (successful DELETE)
- **400** - Bad Request (validation error)
- **404** - Not Found
- **422** - Unprocessable Entity (Pydantic validation error)

## Error Responses

FastAPI automatically returns detailed errors when validation fails:

```json
{
  "detail": [
    {
      "loc": ["body", "price"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

## Best Practices

1. **Separate Input and Output Models**: Use different models for request and response
2. **Use Clear Field Names**: `user_email` is better than `ue`
3. **Add Descriptions**: Help users understand your API
4. **Set Realistic Limits**: Don't allow 10000-character usernames
5. **Use Type Hints**: They power FastAPI's magic

## Next Steps

In Lesson 3, we'll cover:
- More advanced validation techniques
- Custom validators
- Handling file uploads
- Background tasks

---

**Practice Exercise:**
Create a User registration API with these fields:
- username (3-20 characters)
- email (valid email format)
- password (minimum 8 characters)
- age (must be 18 or older)

Try implementing validation for each requirement!