from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Custom schema based on the provided logs
class DetailedLogSummary(BaseModel):
    period: Optional[str] = Field(default=None, description="The time period of the logs")
    total_logs: Optional[int] = Field(default=None, description="Total number of log entries")
    info_count: Optional[int] = Field(default=None, description="Number of INFO entries")
    debug_count: Optional[int] = Field(default=None, description="Number of DEBUG entries")
    error_count: Optional[int] = Field(default=None, description="Number of ERROR entries")
    main_services: Optional[list[str]] = Field(default=None, description="Main services or components involved")
    actions: Optional[list[str]] = Field(default=None, description="Key actions performed in the logs")
    users_involved: Optional[list[str]] = Field(default=None, description="User IDs involved in the logs")
    orders_processed: Optional[list[str]] = Field(default=None, description="Order IDs processed")
    status: Optional[str] = Field(default=None, description="Overall status or outcome of the log sequence")
    performance: Optional[dict] = Field(default=None, description="Performance metrics such as startup time, request duration")
    issues_detected: Optional[list[str]] = Field(default=None, description="Any issues or anomalies detected")
    summary: str = Field(description="A short summary of the logs")

structured_model = model.with_structured_output(DetailedLogSummary)

# Example application log summary (natural style)
logs_summary = """
2025-06-22 10:00:00.001  INFO  [order-service] - Starting OrderServiceApplication v1.0.0...
2025-06-22 10:00:02.143  INFO  [order-service] - Connected to MySQL and Kafka successfully.
2025-06-22 10:00:02.888  INFO  [order-service] - Application started in 2.8 seconds.

2025-06-22 10:02:13.115  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.controller.OrderController - Received POST /orders for userId=12345

2025-06-22 10:02:13.118  DEBUG [order-service] [http-nio-8080-exec-1] c.c.order.validator.OrderValidator - Validating order request: productId=567, quantity=2

2025-06-22 10:02:13.130  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.InventoryClient - Checking inventory for productId=567

2025-06-22 10:02:13.215  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.InventoryClient - Inventory check passed: available=15

2025-06-22 10:02:13.222  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.repository.OrderRepository - Saving order to database...

2025-06-22 10:02:13.248  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.EmailService - Sending confirmation email to userId=12345

2025-06-22 10:02:13.410  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.controller.OrderController - Order placed successfully: orderId=ORD-78910

2025-06-22 10:02:13.411  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.logging.RequestLogger - Completed request for userId=12345 with status=201 duration=296ms
"""

#result = structured_model.invoke(logs_summary)

failure_logs = """2025-06-22 11:15:42.004  INFO  [order-service] [http-nio-8080-exec-5] c.c.order.controller.OrderController - Received POST /orders for userId=12345

2025-06-22 11:15:42.007  DEBUG [order-service] [http-nio-8080-exec-5] c.c.order.validator.OrderValidator - Validating order request: productId=999, quantity=5

2025-06-22 11:15:42.018  INFO  [order-service] [http-nio-8080-exec-5] c.c.order.service.InventoryClient - Checking inventory for productId=999

2025-06-22 11:15:42.099  ERROR [order-service] [http-nio-8080-exec-5] c.c.order.service.InventoryClient - Inventory insufficient for productId=999. Requested=5, Available=2

2025-06-22 11:15:42.101  ERROR [order-service] [http-nio-8080-exec-5] c.c.order.controller.OrderController - Order placement failed for userId=12345: Insufficient inventory
org.example.exception.InventoryException: Not enough inventory for productId=999
	at com.company.order.service.InventoryClient.checkAvailability(InventoryClient.java:52)
	at com.company.order.service.OrderService.placeOrder(OrderService.java:89)
	...

2025-06-22 11:15:42.102  WARN  [order-service] [http-nio-8080-exec-5] c.c.order.logging.RequestLogger - Request failed for userId=12345 with status=400 duration=98ms
"""
result = structured_model.invoke(failure_logs)
print(result)
print("--------------------------------")
if result:
    print(f"Time Range: {result.period}")
    print(f"Total Entries: {result.total_logs}")
    print(f"Main Services: {result.main_services}")
    print(f"Actions: {result.actions}")
    print(f"Status: {result.status}")
    print(f"Summary: {result.summary}")
else:
    print("No structured output returned.") 