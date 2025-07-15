# LangChain Tools and Tool Calling - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Tool Creation](#basic-tool-creation)
3. [Tool Calling with Agents](#tool-calling-with-agents)
4. [Built-in Tools](#built-in-tools)
5. [Custom Tools](#custom-tools)
6. [Advanced Tool Examples](#advanced-tool-examples)
7. [Tool Error Handling](#tool-error-handling)
8. [Best Practices](#best-practices)

## Introduction

LangChain tools are utilities that allow language models to interact with external systems, APIs, databases, and perform various actions. Tools enable agents to:
- Search the web
- Perform calculations
- Access databases
- Call APIs
- Execute code
- And much more

## Basic Tool Creation

### 1. Simple Function-Based Tool

```python
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Basic tool using @tool decorator
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Usage
llm = OpenAI(temperature=0)
tools = [add_numbers]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Test the tool
result = agent.run("What is 15 + 27?")
print(result)
```

### 2. Tool with String Input

```python
@tool
def reverse_string(text: str) -> str:
    """Reverse the input string."""
    return text[::-1]

@tool
def count_words(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())

# Using multiple tools
tools = [reverse_string, count_words]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

result = agent.run("Reverse the string 'Hello World' and count words in 'The quick brown fox'")
print(result)
```

## Tool Calling with Agents

### 3. Basic Agent Setup

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Create a simple calculator tool
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use this for any math calculations.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Setup agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [calculator]

# Create agent
agent = create_react_agent(llm, tools, 
    prompt=PromptTemplate.from_template("""
    You are a helpful assistant. Use the available tools to answer questions.
    
    Tools: {tools}
    
    Question: {input}
    {agent_scratchpad}
    """))

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test
result = agent_executor.invoke({"input": "What is 25 * 4 + 10?"})
print(result)
```

## Built-in Tools

### 4. Web Search Tool

```python
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Initialize search tool
search = DuckDuckGoSearchRun()

# Create agent with search capability
llm = ChatOpenAI(temperature=0)
tools = [search]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Use the search tool
result = agent.run("What is the current price of Bitcoin?")
print(result)
```

### 5. Python REPL Tool

```python
from langchain.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType

# Python execution tool
python_repl = PythonREPLTool()

tools = [python_repl]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Execute Python code
result = agent.run("Create a list of prime numbers up to 20 and calculate their sum")
print(result)
```

## Custom Tools

### 6. Database Query Tool

```python
import sqlite3
from langchain.tools import BaseTool
from typing import Optional
from langchain.callbacks.manager import CallbackManagerForToolRun

class DatabaseQueryTool(BaseTool):
    name = "database_query"
    description = """
    Query a SQLite database. 
    Input should be a valid SQL query.
    Returns the query results.
    """
    
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the SQL query and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            return str(results)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    async def _arun(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version of the tool."""
        raise NotImplementedError("Async not implemented for this tool")

# Usage
db_tool = DatabaseQueryTool("example.db")
tools = [db_tool]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 7. API Call Tool

```python
import requests
from langchain.tools import tool

@tool
def weather_tool(city: str) -> str:
    """
    Get current weather for a city.
    Input should be a city name.
    """
    try:
        # This is a mock API call - replace with actual weather API
        api_key = "your_api_key"
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"Weather in {city}: {temp}°C, {description}"
        else:
            return f"Error fetching weather for {city}"
    except Exception as e:
        return f"Error: {str(e)}"

# Usage
tools = [weather_tool]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
result = agent.run("What's the weather like in New York?")
```

## Advanced Tool Examples

### 8. File System Tool

```python
import os
from langchain.tools import tool
from typing import List

@tool
def list_files(directory: str) -> str:
    """List files in a directory."""
    try:
        files = os.listdir(directory)
        return f"Files in {directory}: {', '.join(files)}"
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return f"Content of {file_path}:\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

# File system tools
file_tools = [list_files, read_file, write_file]
agent = initialize_agent(file_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Test file operations
result = agent.run("List files in current directory, then create a file called 'test.txt' with content 'Hello World'")
```

### 9. Email Tool

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.tools import tool

@tool
def send_email(to_address: str, subject: str, body: str) -> str:
    """
    Send an email.
    Args:
        to_address: Recipient email address
        subject: Email subject
        body: Email body content
    """
    try:
        # Configure your email settings
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        from_address = "your_email@gmail.com"
        password = "your_app_password"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_address, password)
        text = msg.as_string()
        server.sendmail(from_address, to_address, text)
        server.quit()
        
        return f"Email sent successfully to {to_address}"
    except Exception as e:
        return f"Error sending email: {str(e)}"

# Usage
email_tools = [send_email]
agent = initialize_agent(email_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 10. Multi-Step Tool Chain

```python
from langchain.tools import tool
import json

@tool
def data_processor(data: str) -> str:
    """Process JSON data and return statistics."""
    try:
        parsed_data = json.loads(data)
        if isinstance(parsed_data, list):
            return f"List with {len(parsed_data)} items"
        elif isinstance(parsed_data, dict):
            return f"Dictionary with keys: {list(parsed_data.keys())}"
        else:
            return f"Data type: {type(parsed_data)}"
    except Exception as e:
        return f"Error processing data: {str(e)}"

@tool
def data_validator(data: str) -> str:
    """Validate if string is valid JSON."""
    try:
        json.loads(data)
        return "Valid JSON"
    except json.JSONDecodeError:
        return "Invalid JSON"

@tool
def data_formatter(data: str) -> str:
    """Format JSON data with proper indentation."""
    try:
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=2)
        return formatted
    except Exception as e:
        return f"Error formatting data: {str(e)}"

# Chain multiple tools
data_tools = [data_validator, data_processor, data_formatter]
agent = initialize_agent(data_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Test with complex data processing
test_data = '{"users": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]}'
result = agent.run(f"Validate this JSON data, process it, and format it: {test_data}")
```

## Tool Error Handling

### 11. Robust Tool with Error Handling

```python
from langchain.tools import tool
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def robust_calculator(expression: str) -> str:
    """
    A calculator tool with comprehensive error handling.
    Supports basic arithmetic operations (+, -, *, /, **, %).
    """
    try:
        # Validate input
        if not expression or not isinstance(expression, str):
            return "Error: Invalid input. Please provide a mathematical expression."
        
        # Security check - only allow safe operations
        allowed_chars = set('0123456789+-*/.() %')
        if not set(expression.replace(' ', '')).issubset(allowed_chars):
            return "Error: Expression contains invalid characters."
        
        # Check for potential security issues
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return "Error: Expression contains potentially dangerous operations."
        
        # Evaluate expression
        result = eval(expression)
        
        # Log successful calculation
        logger.info(f"Calculated: {expression} = {result}")
        
        return f"Result: {result}"
        
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except SyntaxError:
        return "Error: Invalid mathematical expression syntax."
    except OverflowError:
        return "Error: Number too large to calculate."
    except Exception as e:
        logger.error(f"Unexpected error in calculator: {str(e)}")
        return f"Error: Unable to calculate expression - {str(e)}"

# Usage with error handling
tools = [robust_calculator]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Test various scenarios
test_cases = [
    "2 + 2",
    "10 / 0",
    "invalid_expression",
    "2 ** 1000",
    "import os"
]

for test in test_cases:
    result = agent.run(f"Calculate: {test}")
    print(f"Input: {test}")
    print(f"Result: {result}\n")
```

### 12. Tool with Retry Logic

```python
import time
import random
from langchain.tools import tool

@tool
def api_call_with_retry(endpoint: str, max_retries: int = 3) -> str:
    """
    Make an API call with retry logic.
    Args:
        endpoint: API endpoint to call
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # Simulate API call that might fail
            if random.random() < 0.7:  # 70% chance of failure for demo
                raise Exception("API temporarily unavailable")
            
            # Simulate successful API response
            response = {
                "status": "success",
                "data": f"Data from {endpoint}",
                "timestamp": time.time()
            }
            
            return f"API call successful: {response}"
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                return f"API call failed after {max_retries} attempts: {str(e)}"
    
    return "Unexpected error in retry logic"

# Usage
tools = [api_call_with_retry]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

## Best Practices

### 13. Tool Documentation and Type Hints

```python
from langchain.tools import tool
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

class UserData(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")
    email: str = Field(description="User's email address")

@tool
def process_user_data(
    users: List[Dict[str, Union[str, int]]], 
    filter_age: Optional[int] = None
) -> str:
    """
    Process a list of user data with optional age filtering.
    
    Args:
        users: List of user dictionaries with 'name', 'age', and 'email' keys
        filter_age: Optional minimum age to filter users (default: None)
    
    Returns:
        String summary of processed users
    
    Example:
        users = [
            {"name": "John Doe", "age": 30, "email": "john@example.com"},
            {"name": "Jane Smith", "age": 25, "email": "jane@example.com"}
        ]
    """
    try:
        # Validate input
        if not users or not isinstance(users, list):
            return "Error: Invalid user data provided"
        
        # Filter users by age if specified
        if filter_age is not None:
            filtered_users = [user for user in users if user.get('age', 0) >= filter_age]
        else:
            filtered_users = users
        
        # Process users
        total_users = len(filtered_users)
        avg_age = sum(user.get('age', 0) for user in filtered_users) / total_users if total_users > 0 else 0
        
        # Generate summary
        summary = f"Processed {total_users} users"
        if filter_age:
            summary += f" (filtered by age >= {filter_age})"
        summary += f". Average age: {avg_age:.1f} years"
        
        return summary
        
    except Exception as e:
        return f"Error processing user data: {str(e)}"

# Usage example
tools = [process_user_data]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### 14. Tool Performance Monitoring

```python
import time
from functools import wraps
from langchain.tools import tool

def monitor_performance(func):
    """Decorator to monitor tool performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            print(f"Tool '{func.__name__}' executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Tool '{func.__name__}' failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

@tool
@monitor_performance
def heavy_computation(n: int) -> str:
    """Perform a heavy computation (example: calculate factorial)."""
    if n < 0:
        return "Error: Cannot calculate factorial of negative number"
    
    result = 1
    for i in range(1, n + 1):
        result *= i
        # Simulate heavy computation
        time.sleep(0.01)
    
    return f"Factorial of {n} is {result}"

# Usage
tools = [heavy_computation]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

## Complete Example: Multi-Tool Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import json
import sqlite3
import requests
from datetime import datetime

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define multiple tools
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_to_database(data: str, table_name: str) -> str:
    """Save data to SQLite database."""
    try:
        conn = sqlite3.connect("agent_data.db")
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert data
        cursor.execute(f"INSERT INTO {table_name} (data) VALUES (?)", (data,))
        conn.commit()
        conn.close()
        
        return f"Data saved to {table_name} table successfully"
    except Exception as e:
        return f"Error saving to database: {str(e)}"

@tool
def query_database(query: str) -> str:
    """Execute a SELECT query on the database."""
    try:
        conn = sqlite3.connect("agent_data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return f"Query results: {results}"
    except Exception as e:
        return f"Error querying database: {str(e)}"

@tool
def json_parser(json_string: str) -> str:
    """Parse and format JSON string."""
    try:
        parsed = json.loads(json_string)
        formatted = json.dumps(parsed, indent=2)
        return f"Parsed JSON:\n{formatted}"
    except Exception as e:
        return f"Error parsing JSON: {str(e)}"

# Create agent with all tools
tools = [get_current_time, save_to_database, query_database, json_parser]
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# Test the multi-tool agent
if __name__ == "__main__":
    # Example usage
    test_queries = [
        "What's the current time?",
        "Save this JSON data to a table called 'users': {\"name\": \"John\", \"age\": 30}",
        "Query the users table to show all records",
        "Parse this JSON and format it: {\"products\":[{\"id\":1,\"name\":\"laptop\"}]}"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = agent.run(query)
        print(f"Result: {result}")
        print("=" * 50)
```

## Usage Tips

1. **Tool Names**: Use descriptive names that clearly indicate the tool's purpose
2. **Descriptions**: Write detailed descriptions that help the LLM understand when and how to use the tool
3. **Type Hints**: Use proper type hints for better tool integration
4. **Error Handling**: Always include comprehensive error handling
5. **Validation**: Validate inputs before processing
6. **Security**: Be cautious with tools that execute code or access external systems
7. **Performance**: Monitor tool performance and implement timeouts for long-running operations
8. **Testing**: Test tools thoroughly with various inputs and edge cases

This guide provides a comprehensive overview of LangChain tools, from basic function-based tools to complex multi-tool agents with error handling and performance monitoring.