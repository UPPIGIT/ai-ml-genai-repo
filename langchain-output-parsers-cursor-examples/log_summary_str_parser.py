"""
Application Log Summary Output Parser Example (with Gemini and StrOutputParser)
This example demonstrates how to use LangChain's StrOutputParser to extract just the summary from a log.
"""
from langchain.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

prompt = PromptTemplate.from_template(
    """Read the following application logs and provide a 2-3 sentence summary of what happened.

    Logs:
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
)

parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({})
print("Log summary (plain text):", result) 