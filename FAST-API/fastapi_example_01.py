# 01_hello_world.py
# Your first FastAPI application!

from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Root endpoint - the homepage of your API
@app.get("/")
async def root():
    """
    This is the simplest endpoint.
    Returns a JSON response.
    """
    return {"message": "Hello World"}

# Second endpoint - greet someone by name
@app.get("/hello/{name}")
async def say_hello(name: str):
    """
    Path parameter example.
    Try: http://localhost:8000/hello/John
    """
    return {"message": f"Hello, {name}!"}

# Info endpoint - returns some details about the API
@app.get("/info")
async def get_info():
    """
    Returns information about this API.
    """
    return {
        "app_name": "My First FastAPI App",
        "version": "1.0.0",
        "description": "Learning FastAPI basics"
    }

"""
HOW TO RUN THIS:
1. Save this file as: 01_hello_world.py
2. Open terminal in the same folder
3. Run: uvicorn 01_hello_world:app --reload
4. Open browser: http://localhost:8000
5. See automatic docs: http://localhost:8000/docs

WHAT TO TRY:
- Visit http://localhost:8000
- Visit http://localhost:8000/hello/YourName
- Visit http://localhost:8000/info
- Visit http://localhost:8000/docs (interactive documentation!)
"""