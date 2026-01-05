# 02_basic_operations.py
# Understanding different HTTP methods

from fastapi import FastAPI

app = FastAPI()

# Simulating a simple in-memory database
items_db = {
    1: {"name": "Laptop", "price": 999.99, "stock": 5},
    2: {"name": "Mouse", "price": 29.99, "stock": 50},
    3: {"name": "Keyboard", "price": 79.99, "stock": 30}
}

# GET - Read all items
@app.get("/items")
async def get_all_items():
    """
    GET request to retrieve all items.
    Try: http://localhost:8000/items
    """
    return {"items": items_db, "total": len(items_db)}

# GET - Read single item by ID
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """
    GET request with path parameter.
    Try: http://localhost:8000/items/1
    """
    if item_id in items_db:
        return {"item_id": item_id, "data": items_db[item_id]}
    return {"error": "Item not found"}, 404

# POST - Create new item (we'll improve this in next lesson)
@app.post("/items")
async def create_item(name: str, price: float, stock: int):
    """
    POST request to create a new item.
    For now using query parameters (we'll use request body later).
    Try in docs: http://localhost:8000/docs
    """
    new_id = max(items_db.keys()) + 1
    items_db[new_id] = {"name": name, "price": price, "stock": stock}
    return {"message": "Item created", "item_id": new_id, "data": items_db[new_id]}

# DELETE - Remove an item
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """
    DELETE request to remove an item.
    Try in docs: http://localhost:8000/docs
    """
    if item_id in items_db:
        deleted_item = items_db.pop(item_id)
        return {"message": "Item deleted", "deleted_item": deleted_item}
    return {"error": "Item not found"}

# GET - Statistics endpoint
@app.get("/stats")
async def get_statistics():
    """
    Calculate some statistics from our items.
    """
    total_items = len(items_db)
    total_value = sum(item["price"] * item["stock"] for item in items_db.values())
    total_stock = sum(item["stock"] for item in items_db.values())
    
    return {
        "total_items": total_items,
        "total_inventory_value": round(total_value, 2),
        "total_stock_units": total_stock
    }

"""
HOW TO RUN:
uvicorn 02_basic_operations:app --reload

WHAT TO TRY:
1. GET all items: http://localhost:8000/items
2. GET single item: http://localhost:8000/items/1
3. View stats: http://localhost:8000/stats
4. Go to http://localhost:8000/docs and try POST/DELETE operations

IMPORTANT NOTE:
This uses an in-memory database, so data resets when you restart the server.
We'll add real database persistence in later lessons!
"""