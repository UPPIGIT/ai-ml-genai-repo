# 04_request_body_basics.py
# Introduction to Pydantic models and request body handling

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Define a Pydantic model (schema) for a Product
class Product(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    quantity: int
    is_available: bool = True

# In-memory storage
products_db = []
product_id_counter = 1

@app.get("/")
async def root():
    """Homepage with API info"""
    return {
        "message": "Product API with Request Body validation",
        "endpoints": {
            "POST /products": "Create a product",
            "GET /products": "List all products",
            "GET /products/{id}": "Get specific product",
            "PUT /products/{id}": "Update a product"
        }
    }

@app.post("/products")
async def create_product(product: Product):
    """
    Create a new product using request body.
    The 'product: Product' parameter tells FastAPI to:
    1. Expect JSON in the request body
    2. Validate it against the Product model
    3. Convert it to a Product object
    
    Test in http://localhost:8000/docs with this JSON:
    {
        "name": "Laptop",
        "description": "High-performance laptop",
        "price": 999.99,
        "quantity": 10,
        "is_available": true
    }
    """
    global product_id_counter
    
    # Convert Pydantic model to dict and add ID
    product_dict = product.dict()
    product_dict["id"] = product_id_counter
    product_id_counter += 1
    
    products_db.append(product_dict)
    
    return {
        "message": "Product created successfully",
        "product": product_dict
    }

@app.get("/products")
async def get_all_products():
    """Get all products"""
    return {
        "total": len(products_db),
        "products": products_db
    }

@app.get("/products/{product_id}")
async def get_product(product_id: int):
    """Get a specific product by ID"""
    product = next((p for p in products_db if p["id"] == product_id), None)
    
    if product:
        return {"product": product}
    
    return {"error": "Product not found"}, 404

@app.put("/products/{product_id}")
async def update_product(product_id: int, product: Product):
    """
    Update an existing product completely.
    Combines path parameter (product_id) with request body (product).
    
    Test with:
    PUT http://localhost:8000/products/1
    {
        "name": "Updated Laptop",
        "description": "Now with more RAM",
        "price": 1099.99,
        "quantity": 8,
        "is_available": true
    }
    """
    # Find the product
    for i, p in enumerate(products_db):
        if p["id"] == product_id:
            # Update with new data
            updated_product = product.dict()
            updated_product["id"] = product_id
            products_db[i] = updated_product
            
            return {
                "message": "Product updated successfully",
                "product": updated_product
            }
    
    return {"error": "Product not found"}, 404

@app.delete("/products/{product_id}")
async def delete_product(product_id: int):
    """Delete a product by ID"""
    for i, p in enumerate(products_db):
        if p["id"] == product_id:
            deleted = products_db.pop(i)
            return {
                "message": "Product deleted successfully",
                "deleted_product": deleted
            }
    
    return {"error": "Product not found"}, 404

# Additional model example with more fields
class ProductStats(BaseModel):
    total_products: int
    total_value: float
    in_stock: int
    out_of_stock: int

@app.get("/products/stats/summary", response_model=ProductStats)
async def get_product_stats():
    """
    Returns statistics with a defined response model.
    response_model ensures the response matches ProductStats structure.
    """
    in_stock = sum(1 for p in products_db if p["is_available"])
    total_value = sum(p["price"] * p["quantity"] for p in products_db)
    
    return ProductStats(
        total_products=len(products_db),
        total_value=round(total_value, 2),
        in_stock=in_stock,
        out_of_stock=len(products_db) - in_stock
    )

"""
HOW TO RUN:
uvicorn 04_request_body_basics:app --reload

KEY CONCEPTS DEMONSTRATED:
1. Pydantic BaseModel for data structure
2. Request body validation (FastAPI does this automatically!)
3. Optional fields with defaults
4. Combining path parameters with request body
5. Response models for structured output

WHAT TO TRY:
1. Go to http://localhost:8000/docs
2. Try the POST /products endpoint with valid data
3. Try with INVALID data (e.g., price as "abc" instead of number)
4. See how FastAPI automatically returns validation errors!
5. Create several products, then check stats

VALIDATION HAPPENS AUTOMATICALLY:
- If you send "price": "not_a_number", FastAPI returns an error
- If you miss required fields, FastAPI returns an error
- If you send extra fields, FastAPI ignores them (by default)
"""