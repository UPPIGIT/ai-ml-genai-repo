# 06_nested_models.py
# Working with nested Pydantic models (models within models)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

app = FastAPI(title="Nested Models API - E-commerce Example")

# Enums
class OrderStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    shipped = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"

class PaymentMethod(str, Enum):
    credit_card = "credit_card"
    debit_card = "debit_card"
    paypal = "paypal"
    cash = "cash"

# Nested model 1: Address
class Address(BaseModel):
    """Address information - will be nested in other models"""
    street: str = Field(..., min_length=5, max_length=100)
    city: str = Field(..., min_length=2, max_length=50)
    state: str = Field(..., min_length=2, max_length=50)
    postal_code: str = Field(..., min_length=5, max_length=10)
    country: str = Field(..., min_length=2, max_length=50)
    
    class Config:
        schema_extra = {
            "example": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA"
            }
        }

# Nested model 2: Order Item
class OrderItem(BaseModel):
    """Individual item in an order"""
    product_id: int = Field(..., gt=0)
    product_name: str
    quantity: int = Field(..., gt=0, le=100)
    price_per_unit: float = Field(..., gt=0)
    
    @property
    def total_price(self) -> float:
        """Calculate total price for this item"""
        return self.quantity * self.price_per_unit

# Nested model 3: Payment Info
class PaymentInfo(BaseModel):
    """Payment information"""
    method: PaymentMethod
    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0)
    paid_at: Optional[datetime] = None

# Main model: Customer (contains Address)
class Customer(BaseModel):
    """Customer with embedded address"""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=15)
    address: Address  # Nested model!
    
    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "1234567890",
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "state": "NY",
                    "postal_code": "10001",
                    "country": "USA"
                }
            }
        }

# Main model: Order (contains multiple nested models)
class Order(BaseModel):
    """Order with nested customer, items, and payment info"""
    customer: Customer  # Nested Customer model
    items: List[OrderItem] = Field(..., min_items=1)  # List of nested OrderItem models
    payment: PaymentInfo  # Nested PaymentInfo model
    status: OrderStatus = OrderStatus.pending
    notes: Optional[str] = Field(None, max_length=500)
    
    @property
    def total_amount(self) -> float:
        """Calculate total order amount"""
        return sum(item.quantity * item.price_per_unit for item in self.items)

# Response model for created order
class OrderResponse(BaseModel):
    """Response after creating an order"""
    order_id: int
    customer_name: str
    customer_email: EmailStr
    total_items: int
    total_amount: float
    status: OrderStatus
    created_at: datetime

# In-memory storage
orders_db = []
order_counter = 1

@app.post("/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(order: Order):
    """
    Create a new order with nested customer, items, and payment data.
    
    Example request body:
    {
        "customer": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA"
            }
        },
        "items": [
            {
                "product_id": 1,
                "product_name": "Laptop",
                "quantity": 2,
                "price_per_unit": 999.99
            },
            {
                "product_id": 2,
                "product_name": "Mouse",
                "quantity": 1,
                "price_per_unit": 29.99
            }
        ],
        "payment": {
            "method": "credit_card",
            "transaction_id": "txn_123456",
            "amount": 2029.97,
            "paid_at": "2024-01-15T10:30:00"
        },
        "status": "pending",
        "notes": "Please deliver before 5 PM"
    }
    """
    global order_counter
    
    # Validate payment amount matches order total
    calculated_total = order.total_amount
    if abs(order.payment.amount - calculated_total) > 0.01:  # Allow tiny floating point difference
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment amount {order.payment.amount} doesn't match order total {calculated_total}"
        )
    
    # Store order
    order_dict = order.dict()
    order_dict["order_id"] = order_counter
    order_dict["created_at"] = datetime.now()
    orders_db.append(order_dict)
    
    order_counter += 1
    
    # Return simplified response
    return OrderResponse(
        order_id=order_dict["order_id"],
        customer_name=order.customer.name,
        customer_email=order.customer.email,
        total_items=len(order.items),
        total_amount=calculated_total,
        status=order.status,
        created_at=order_dict["created_at"]
    )

@app.get("/orders/{order_id}")
async def get_order(order_id: int):
    """Get complete order details by ID"""
    order = next((o for o in orders_db if o["order_id"] == order_id), None)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return order

@app.get("/orders")
async def get_all_orders(status: Optional[OrderStatus] = None):
    """
    Get all orders, optionally filtered by status.
    Status can be: pending, processing, shipped, delivered, cancelled
    """
    if status:
        filtered = [o for o in orders_db if o["status"] == status]
        return {
            "status_filter": status,
            "count": len(filtered),
            "orders": filtered
        }
    
    return {
        "total": len(orders_db),
        "orders": orders_db
    }

@app.patch("/orders/{order_id}/status")
async def update_order_status(order_id: int, new_status: OrderStatus):
    """
    Update order status.
    Send just the new status in the request body: "processing"
    """
    order = next((o for o in orders_db if o["order_id"] == order_id), None)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    old_status = order["status"]
    order["status"] = new_status
    
    return {
        "message": "Status updated",
        "order_id": order_id,
        "old_status": old_status,
        "new_status": new_status
    }

@app.get("/customers/{email}/orders")
async def get_customer_orders(email: EmailStr):
    """Get all orders for a specific customer by email"""
    customer_orders = [
        o for o in orders_db 
        if o["customer"]["email"] == email
    ]
    
    if not customer_orders:
        return {
            "message": f"No orders found for {email}",
            "orders": []
        }
    
    return {
        "customer_email": email,
        "total_orders": len(customer_orders),
        "orders": customer_orders
    }

"""
HOW TO RUN:
uvicorn 06_nested_models:app --reload

KEY CONCEPTS DEMONSTRATED:
1. Nested models (models inside models)
2. Lists of models (multiple items in an order)
3. Complex validation across nested structures
4. Property methods for calculated fields
5. Real-world e-commerce example

NESTED STRUCTURE:
Order
├── Customer
│   └── Address
├── Items (List)
│   └── OrderItem (multiple)
└── Payment
    └── PaymentInfo

WHAT TO TRY:
1. Go to http://localhost:8000/docs
2. Create an order with the example JSON
3. Try with mismatched payment amount (validation!)
4. Create multiple orders
5. Filter orders by status
6. Search orders by customer email

BENEFITS OF NESTED MODELS:
- Clear, organized data structure
- Automatic validation at every level
- Reusable components (Address can be used anywhere)
- Self-documenting API
"""