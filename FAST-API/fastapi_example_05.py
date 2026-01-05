# 05_advanced_validation.py
# Advanced Pydantic validation with Field constraints

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional
from enum import Enum
from datetime import datetime

app = FastAPI(title="Advanced Validation API")

# Enum for fixed choices
class UserRole(str, Enum):
    admin = "admin"
    user = "user"
    guest = "guest"

class Category(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    food = "food"
    books = "books"

# Model with Field validation
class User(BaseModel):
    username: str = Field(
        ..., 
        min_length=3, 
        max_length=20,
        description="Username must be 3-20 characters"
    )
    email: EmailStr = Field(
        ...,
        description="Valid email address required"
    )
    password: str = Field(
        ...,
        min_length=8,
        description="Password must be at least 8 characters"
    )
    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Age must be between 18 and 120"
    )
    role: UserRole = Field(
        default=UserRole.user,
        description="User role in the system"
    )
    
    # Custom validator
    @validator('username')
    def username_alphanumeric(cls, v):
        """Ensure username contains only letters and numbers"""
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        """Check password contains at least one number"""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one number')
        return v

# Response model (excludes sensitive data)
class UserResponse(BaseModel):
    username: str
    email: EmailStr
    age: int
    role: UserRole
    created_at: datetime
    
    class Config:
        # Example data for documentation
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 25,
                "role": "user",
                "created_at": "2024-01-15T10:30:00"
            }
        }

# Product model with extensive validation
class Product(BaseModel):
    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Product name"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Product description (optional)"
    )
    price: float = Field(
        ...,
        gt=0,
        lt=1000000,
        description="Price must be greater than 0"
    )
    category: Category = Field(
        ...,
        description="Product category"
    )
    quantity: int = Field(
        default=0,
        ge=0,
        description="Quantity in stock (cannot be negative)"
    )
    discount_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Discount percentage (0-100)"
    )
    tags: list[str] = Field(
        default=[],
        description="Product tags"
    )
    
    @validator('tags')
    def limit_tags(cls, v):
        """Limit number of tags to 5"""
        if len(v) > 5:
            raise ValueError('Maximum 5 tags allowed')
        return v
    
    @validator('discount_percent')
    def validate_discount(cls, v, values):
        """Ensure discounted price is still positive"""
        if v is not None and 'price' in values:
            discounted_price = values['price'] * (1 - v / 100)
            if discounted_price <= 0:
                raise ValueError('Discount too high, price would be zero or negative')
        return v

# Partial update model (all fields optional)
class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    price: Optional[float] = Field(None, gt=0, lt=1000000)
    category: Optional[Category] = None
    quantity: Optional[int] = Field(None, ge=0)
    discount_percent: Optional[float] = Field(None, ge=0, le=100)
    tags: Optional[list[str]] = None

# In-memory storage
users_db = []
products_db = []

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: User):
    """
    Create a new user with validation.
    Password is NOT returned in response (security!).
    
    Try this valid example:
    {
        "username": "alice123",
        "email": "alice@example.com",
        "password": "secure123",
        "age": 25,
        "role": "user"
    }
    
    Try these invalid examples to see validation:
    - username with special chars: "alice@123"
    - password without number: "password"
    - age under 18: 16
    - invalid email: "not-an-email"
    """
    # Check if username already exists
    if any(u["username"] == user.username for u in users_db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Store user (in real app, hash the password!)
    user_dict = user.dict()
    user_dict["created_at"] = datetime.now()
    users_db.append(user_dict)
    
    # Return without password
    return UserResponse(**user_dict)

@app.get("/users")
async def get_users():
    """Get all users (passwords excluded in real scenario)"""
    return {
        "total": len(users_db),
        "users": [
            {k: v for k, v in u.items() if k != "password"}
            for u in users_db
        ]
    }

@app.post("/products", status_code=status.HTTP_201_CREATED)
async def create_product(product: Product):
    """
    Create a product with extensive validation.
    
    Valid example:
    {
        "name": "Laptop Pro",
        "description": "High-performance laptop",
        "price": 1299.99,
        "category": "electronics",
        "quantity": 10,
        "discount_percent": 15,
        "tags": ["tech", "computer", "premium"]
    }
    
    Try invalid data:
    - negative price
    - more than 5 tags
    - discount that makes price <= 0
    """
    product_dict = product.dict()
    product_dict["id"] = len(products_db) + 1
    product_dict["created_at"] = datetime.now()
    products_db.append(product_dict)
    
    return {
        "message": "Product created",
        "product": product_dict
    }

@app.patch("/products/{product_id}")
async def update_product_partial(product_id: int, update: ProductUpdate):
    """
    Partially update a product (PATCH = partial update).
    Only send fields you want to change!
    
    Example - only update price:
    {
        "price": 1099.99
    }
    """
    # Find product
    product = next((p for p in products_db if p["id"] == product_id), None)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Update only provided fields
    update_data = update.dict(exclude_unset=True)  # Only include fields that were set
    for key, value in update_data.items():
        product[key] = value
    
    return {
        "message": "Product updated",
        "product": product
    }

@app.get("/products")
async def get_products(category: Optional[Category] = None):
    """
    Get products, optionally filtered by category.
    Category must be one of: electronics, clothing, food, books
    """
    if category:
        filtered = [p for p in products_db if p["category"] == category]
        return {"category": category, "products": filtered}
    
    return {"products": products_db}

"""
HOW TO RUN:
uvicorn 05_advanced_validation:app --reload

KEY CONCEPTS:
1. Field() for detailed validation rules
2. EmailStr for email validation
3. Enums for fixed choices
4. Custom validators with @validator
5. Response models to hide sensitive data
6. Partial updates with PATCH
7. HTTP status codes (201 for created, 404 for not found)

WHAT TO TRY:
1. Go to http://localhost:8000/docs
2. Create a user with VALID data
3. Try to create user with INVALID data (see detailed errors!)
4. Create products with different categories
5. Try partial updates with PATCH

VALIDATION ERRORS:
FastAPI returns clear error messages showing:
- Which field failed
- Why it failed
- What the requirements are
"""