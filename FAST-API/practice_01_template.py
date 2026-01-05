# practice_01_exercise_template.py
# Exercise 1: Blog API - YOUR TURN TO CODE!

"""
EXERCISE GOAL:
Build a simple blog API with posts and validation.

REQUIREMENTS:
1. BlogPost Model with:
   - title (string, 5-100 characters, required)
   - content (string, 50-5000 characters, required)
   - author (string, 3-50 characters, required)
   - tags (list of strings, optional, max 5 tags)
   - published (boolean, default False)

2. Implement these endpoints:
   - POST /posts - Create a new blog post
   - GET /posts - Get all posts
   - GET /posts/{post_id} - Get specific post
   - PUT /posts/{post_id} - Update entire post
   - DELETE /posts/{post_id} - Delete a post
   - GET /posts/published - Get only published posts

3. BONUS CHALLENGES:
   - Add custom validator to ensure tags are lowercase
   - Add endpoint to search posts by tag
   - Add pagination (skip and limit) to GET /posts

HINTS:
- Use Field() for validation constraints
- Use in-memory list for storage (we'll add database later)
- Don't forget to import everything you need!
- Test using http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(title="Blog API - Exercise 1")

# ============================================================================
# TODO 1: Create your BlogPost model here
# ============================================================================

class BlogPost(BaseModel):
    # YOUR CODE HERE
    # Remember to add Field() with validation constraints!
    pass


# ============================================================================
# TODO 2: Create in-memory storage
# ============================================================================

# YOUR CODE HERE
# Hint: You'll need a list to store posts and a counter for IDs


# ============================================================================
# TODO 3: Implement POST /posts endpoint
# ============================================================================

@app.post("/posts")
async def create_post(post: BlogPost):
    """
    Create a new blog post.
    
    TODO:
    - Generate a unique post_id
    - Add created_at timestamp
    - Store in database
    - Return the created post
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TODO 4: Implement GET /posts endpoint
# ============================================================================

@app.get("/posts")
async def get_all_posts():
    """
    Get all blog posts.
    
    TODO:
    - Return all posts from storage
    - Include total count
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TODO 5: Implement GET /posts/{post_id} endpoint
# ============================================================================

@app.get("/posts/{post_id}")
async def get_post(post_id: int):
    """
    Get a specific post by ID.
    
    TODO:
    - Find post by post_id
    - Return 404 if not found
    - Return the post if found
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TODO 6: Implement PUT /posts/{post_id} endpoint
# ============================================================================

@app.put("/posts/{post_id}")
async def update_post(post_id: int, post: BlogPost):
    """
    Update an entire post.
    
    TODO:
    - Find the post by ID
    - Replace all fields with new data
    - Keep the original post_id and created_at
    - Add/update updated_at timestamp
    - Return 404 if post not found
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TODO 7: Implement DELETE /posts/{post_id} endpoint
# ============================================================================

@app.delete("/posts/{post_id}")
async def delete_post(post_id: int):
    """
    Delete a post.
    
    TODO:
    - Find and remove post from storage
    - Return success message or 404 if not found
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TODO 8: Implement GET /posts/published endpoint
# ============================================================================

@app.get("/posts/published")
async def get_published_posts():
    """
    Get only published posts.
    
    TODO:
    - Filter posts where published = True
    - Return the filtered list
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# BONUS TODO 9: Add custom validator for tags (make them lowercase)
# ============================================================================

# HINT: Use @validator decorator in your BlogPost model


# ============================================================================
# BONUS TODO 10: Add search by tag endpoint
# ============================================================================

@app.get("/posts/search")
async def search_by_tag(tag: str):
    """
    Search posts by tag.
    
    TODO:
    - Take tag as query parameter
    - Find all posts that have this tag
    - Return matching posts
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# BONUS TODO 11: Add pagination to GET /posts
# ============================================================================

# HINT: Add skip and limit as query parameters


"""
===============================================================================
TESTING CHECKLIST:
===============================================================================

✅ Step 1: Create a valid post
   - Go to http://localhost:8000/docs
   - Try POST /posts with valid data
   - Check if it returns the created post with ID

✅ Step 2: Test validation
   - Try creating post with title too short (< 5 chars)
   - Try creating post with content too short (< 50 chars)
   - Try creating post with more than 5 tags
   - Check that you get validation errors

✅ Step 3: Test GET endpoints
   - Create 2-3 posts
   - Get all posts
   - Get specific post by ID
   - Try to get non-existent post (should return error)

✅ Step 4: Test UPDATE
   - Update a post with PUT
   - Verify the changes

✅ Step 5: Test DELETE
   - Delete a post
   - Try to get it again (should not exist)

✅ Step 6: Test published filter
   - Create some published and unpublished posts
   - Get only published posts
   - Verify unpublished ones are not returned

✅ BONUS: Test search by tag
   - Create posts with different tags
   - Search for a specific tag
   - Verify only matching posts are returned

===============================================================================
EXAMPLE TEST DATA:
===============================================================================

Valid Post:
{
  "title": "My First Blog Post",
  "content": "This is the content of my blog post. It needs to be at least 50 characters long to pass validation.",
  "author": "John Doe",
  "tags": ["python", "fastapi"],
  "published": true
}

Invalid Post (title too short):
{
  "title": "Hi",
  "content": "This is the content of my blog post. It needs to be at least 50 characters long.",
  "author": "John Doe",
  "tags": ["python"],
  "published": false
}

===============================================================================
HOW TO RUN:
===============================================================================
uvicorn practice_01_exercise_template:app --reload

Then open: http://localhost:8000/docs

===============================================================================
NEED HELP?
===============================================================================

If you get stuck on any TODO:
1. Look at the example files (04_request_body_basics.py, 05_advanced_validation.py)
2. Check the Lesson 2 markdown guide
3. Try to solve one endpoint at a time
4. Test each endpoint before moving to the next

COMMON ISSUES:
- Forgot to import something? Check imports at the top
- Validation not working? Make sure you used Field() correctly
- Can't find post? Remember list indexing starts at 0
- 404 errors? Make sure you're raising HTTPException

Good luck! 🚀
"""