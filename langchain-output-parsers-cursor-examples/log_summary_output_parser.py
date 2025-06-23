"""
Application Log Summary Output Parser Example (with Gemini and PydanticOutputParser)
This example demonstrates how to use LangChain's PydanticOutputParser to parse a log summary into a structured Pydantic model.
"""
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

# Log summary schema
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

# Prompt for the log summary
prompt = PromptTemplate.from_template(
    """Extract the following fields from these application logs and return as structured data:\n
    - period
    - total_logs
    - info_count
    - debug_count
    - error_count
    - main_services
    - actions
    - users_involved
    - orders_processed
    - status
    - performance
    - issues_detected
    - summary\n\nLogs:\n2025-06-22 10:00:00.001  INFO  [order-service] - Starting OrderServiceApplication v1.0.0...\n2025-06-22 10:00:02.143  INFO  [order-service] - Connected to MySQL and Kafka successfully.\n2025-06-22 10:00:02.888  INFO  [order-service] - Application started in 2.8 seconds.\n2025-06-22 10:02:13.115  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.controller.OrderController - Received POST /orders for userId=12345\n2025-06-22 10:02:13.118  DEBUG [order-service] [http-nio-8080-exec-1] c.c.order.validator.OrderValidator - Validating order request: productId=567, quantity=2\n2025-06-22 10:02:13.130  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.InventoryClient - Checking inventory for productId=567\n2025-06-22 10:02:13.215  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.InventoryClient - Inventory check passed: available=15\n2025-06-22 10:02:13.222  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.repository.OrderRepository - Saving order to database...\n2025-06-22 10:02:13.248  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.service.EmailService - Sending confirmation email to userId=12345\n2025-06-22 10:02:13.410  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.controller.OrderController - Order placed successfully: orderId=ORD-78910\n2025-06-22 10:02:13.411  INFO  [order-service] [http-nio-8080-exec-1] c.c.order.logging.RequestLogger - Completed request for userId=12345 with status=201 duration=296ms\n"""
)

parser = PydanticOutputParser(pydantic_object=DetailedLogSummary)
chain = prompt | model | parser

result = chain.invoke({})
print("Parsed log summary:", result) 