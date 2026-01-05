# 03_query_parameters.py
# Working with query parameters and optional parameters

from fastapi import FastAPI
from typing import Optional

app = FastAPI()

# Sample products database
products = [
    {"id": 1, "name": "Laptop", "category": "electronics", "price": 999.99, "in_stock": True},
    {"id": 2, "name": "Mouse", "category": "electronics", "price": 29.99, "in_stock": True},
    {"id": 3, "name": "Desk", "category": "furniture", "price": 299.99, "in_stock": False},
    {"id": 4, "name": "Chair", "category": "furniture", "price": 199.99, "in_stock": True},
    {"id": 5, "name": "Monitor", "category": "electronics", "price": 349.99, "in_stock": True},
    {"id": 6, "name": "Lamp", "category": "furniture", "price": 49.99, "in_stock": True},
]

# Basic query parameters
@app.get("/products")
async def get_products(skip: int = 0, limit: int = 10):
    """
    Query parameters with default values.
    skip: how many items to skip (pagination)
    limit: maximum items to return
    
    Try:
    - http://localhost:8000/products
    - http://localhost:8000/products?skip=2
    - http://localhost:8000/products?skip=1&limit=3
    """
    return {
        "skip": skip,
        "limit": limit,
        "products": products[skip:skip + limit],
        "total": len(products)
    }

# Optional query parameters
@app.get("/search")
async def search_products(
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock: Optional[bool] = None
):
    """
    Filter products using optional query parameters.
    
    Try:
    - http://localhost:8000/search?category=electronics
    - http://localhost:8000/search?min_price=50&max_price=300
    - http://localhost:8000/search?in_stock=true
    - http://localhost:8000/search?category=furniture&in_stock=true
    """
    filtered_products = products.copy()
    
    # Filter by category
    if category:
        filtered_products = [p for p in filtered_products if p["category"] == category]
    
    # Filter by price range
    if min_price is not None:
        filtered_products = [p for p in filtered_products if p["price"] >= min_price]
    if max_price is not None:
        filtered_products = [p for p in filtered_products if p["price"] <= max_price]
    
    # Filter by stock status
    if in_stock is not None:
        filtered_products = [p for p in filtered_products if p["in_stock"] == in_stock]
    
    return {
        "filters": {
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "in_stock": in_stock
        },
        "results": filtered_products,
        "count": len(filtered_products)
    }

# Combining path and query parameters
@app.get("/products/{product_id}/details")
async def get_product_details(product_id: int, include_similar: bool = False):
    """
    Mix of path parameter (product_id) and query parameter (include_similar).
    
    Try:
    - http://localhost:8000/products/1/details
    - http://localhost:8000/products/1/details?include_similar=true
    """
    product = next((p for p in products if p["id"] == product_id), None)
    
    if not product:
        return {"error": "Product not found"}
    
    result = {"product": product}
    
    if include_similar:
        similar = [
            p for p in products 
            if p["category"] == product["category"] and p["id"] != product_id
        ]
        result["similar_products"] = similar
    
    return result

# Query parameter with multiple values (list)
@app.get("/products/by-ids")
async def get_products_by_ids(ids: list[int] = []):
    """
    Query parameter that accepts multiple values.
    
    Try:
    - http://localhost:8000/products/by-ids?ids=1&ids=3&ids=5
    """
    selected_products = [p for p in products if p["id"] in ids]
    return {
        "requested_ids": ids,
        "products": selected_products,
        "found": len(selected_products)
    }

"""
HOW TO RUN:
uvicorn 03_query_parameters:app --reload

KEY CONCEPTS:
1. Query Parameters come after ? in URL: /path?param1=value1&param2=value2
2. Default values make parameters optional
3. Optional[type] = None makes parameters truly optional
4. You can combine path parameters and query parameters
5. FastAPI automatically validates types

WHAT TO TRY:
1. Visit http://localhost:8000/docs to see interactive documentation
2. Try different combinations of query parameters
3. Try invalid values (e.g., text instead of number) to see automatic validation
"""