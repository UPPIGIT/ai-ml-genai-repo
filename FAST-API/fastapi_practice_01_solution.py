# practice_01_blog_api.py
# Exercise 1: Blog API - Complete Solution with Comments

from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="Blog API", description="Practice Exercise 1")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BlogPost(BaseModel):
    """
    Blog post model with validation rules.
    """
    title: str = Field(
        ..., 
        min_length=5, 
        max_length=100,
        description="Post title must be 5-100 characters"
    )
    content: str = Field(
        ...,
        min_length=50,
        max_length=5000,
        description="Post content must be 50-5000 characters"
    )
    author: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Author name must be 3-50 characters"
    )
    tags: Optional[List[str]] = Field(
        default=[],
        max_items=5,
        description="Maximum 5 tags allowed"
    )
    published: bool = Field(
        default=False,
        description="Whether the post is published"
    )
    
    # Custom validator: ensure tags are lowercase
    @validator('tags')
    def tags_must_be_lowercase(cls, v):
        """Convert all tags to lowercase and ensure no duplicates"""
        if v:
            # Convert to lowercase
            lowercase_tags = [tag.lower().strip() for tag in v]
            # Remove duplicates while preserving order
            seen = set()
            unique_tags = []
            for tag in lowercase_tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)
            return unique_tags
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "title": "My First Blog Post",
                "content": "This is the content of my blog post. It needs to be at least 50 characters long to pass validation.",
                "author": "John Doe",
                "tags": ["python", "fastapi", "tutorial"],
                "published": False
            }
        }

class BlogPostResponse(BaseModel):
    """Response model including the generated post_id and created_at"""
    post_id: int
    title: str
    content: str
    author: str
    tags: List[str]
    published: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

class BlogPostUpdate(BaseModel):
    """Model for updating a blog post - all fields optional"""
    title: Optional[str] = Field(None, min_length=5, max_length=100)
    content: Optional[str] = Field(None, min_length=50, max_length=5000)
    author: Optional[str] = Field(None, min_length=3, max_length=50)
    tags: Optional[List[str]] = Field(None, max_items=5)
    published: Optional[bool] = None

# ============================================================================
# IN-MEMORY DATABASE
# ============================================================================

posts_db = []
post_counter = 1

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Blog API - Practice Exercise 1",
        "endpoints": {
            "POST /posts": "Create a new blog post",
            "GET /posts": "Get all posts with pagination",
            "GET /posts/{post_id}": "Get specific post",
            "PUT /posts/{post_id}": "Update entire post",
            "PATCH /posts/{post_id}": "Partially update post",
            "DELETE /posts/{post_id}": "Delete a post",
            "GET /posts/published/only": "Get only published posts",
            "GET /posts/search/by-tag": "Search posts by tag"
        },
        "total_posts": len(posts_db)
    }

@app.post("/posts", response_model=BlogPostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(post: BlogPost):
    """
    Create a new blog post.
    
    Validation rules:
    - title: 5-100 characters
    - content: 50-5000 characters
    - author: 3-50 characters
    - tags: maximum 5, will be converted to lowercase
    - published: default False
    """
    global post_counter
    
    # Create post dictionary with additional fields
    post_dict = post.dict()
    post_dict["post_id"] = post_counter
    post_dict["created_at"] = datetime.now()
    post_dict["updated_at"] = None
    
    # Add to database
    posts_db.append(post_dict)
    post_counter += 1
    
    return BlogPostResponse(**post_dict)

@app.get("/posts", response_model=List[BlogPostResponse])
async def get_all_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum posts to return")
):
    """
    Get all blog posts with pagination.
    
    Query parameters:
    - skip: Number of posts to skip (default: 0)
    - limit: Maximum number of posts to return (default: 10, max: 100)
    """
    return [BlogPostResponse(**post) for post in posts_db[skip:skip + limit]]

@app.get("/posts/{post_id}", response_model=BlogPostResponse)
async def get_post(post_id: int):
    """Get a specific blog post by ID"""
    post = next((p for p in posts_db if p["post_id"] == post_id), None)
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id {post_id} not found"
        )
    
    return BlogPostResponse(**post)

@app.put("/posts/{post_id}", response_model=BlogPostResponse)
async def update_post_full(post_id: int, post: BlogPost):
    """
    Update an entire blog post (replaces all fields).
    All fields must be provided.
    """
    # Find the post
    for i, p in enumerate(posts_db):
        if p["post_id"] == post_id:
            # Update with new data, keeping post_id and created_at
            updated_post = post.dict()
            updated_post["post_id"] = post_id
            updated_post["created_at"] = p["created_at"]
            updated_post["updated_at"] = datetime.now()
            
            posts_db[i] = updated_post
            return BlogPostResponse(**updated_post)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with id {post_id} not found"
    )

@app.patch("/posts/{post_id}", response_model=BlogPostResponse)
async def update_post_partial(post_id: int, post_update: BlogPostUpdate):
    """
    Partially update a blog post.
    Only send the fields you want to change.
    """
    # Find the post
    for i, p in enumerate(posts_db):
        if p["post_id"] == post_id:
            # Update only provided fields
            update_data = post_update.dict(exclude_unset=True)
            
            for field, value in update_data.items():
                p[field] = value
            
            p["updated_at"] = datetime.now()
            
            return BlogPostResponse(**p)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with id {post_id} not found"
    )

@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(post_id: int):
    """Delete a blog post"""
    for i, p in enumerate(posts_db):
        if p["post_id"] == post_id:
            posts_db.pop(i)
            return  # 204 returns no content
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Post with id {post_id} not found"
    )

@app.get("/posts/published/only", response_model=List[BlogPostResponse])
async def get_published_posts():
    """Get only published blog posts"""
    published = [p for p in posts_db if p["published"]]
    
    return [BlogPostResponse(**post) for post in published]

@app.get("/posts/search/by-tag", response_model=List[BlogPostResponse])
async def search_by_tag(tag: str = Query(..., description="Tag to search for")):
    """
    Search posts by tag.
    Tag search is case-insensitive.
    
    Example: /posts/search/by-tag?tag=python
    """
    tag_lower = tag.lower()
    matching_posts = [
        p for p in posts_db 
        if any(t.lower() == tag_lower for t in p["tags"])
    ]
    
    if not matching_posts:
        return []
    
    return [BlogPostResponse(**post) for post in matching_posts]

@app.get("/stats")
async def get_stats():
    """Get blog statistics"""
    total = len(posts_db)
    published = sum(1 for p in posts_db if p["published"])
    unpublished = total - published
    
    # Count posts per author
    authors = {}
    for post in posts_db:
        author = post["author"]
        authors[author] = authors.get(author, 0) + 1
    
    # Get all unique tags
    all_tags = set()
    for post in posts_db:
        all_tags.update(post["tags"])
    
    return {
        "total_posts": total,
        "published_posts": published,
        "unpublished_posts": unpublished,
        "total_authors": len(authors),
        "posts_per_author": authors,
        "unique_tags": sorted(list(all_tags)),
        "total_unique_tags": len(all_tags)
    }

# ============================================================================
# BONUS: Author-specific endpoints
# ============================================================================

@app.get("/authors/{author_name}/posts", response_model=List[BlogPostResponse])
async def get_posts_by_author(author_name: str):
    """Get all posts by a specific author"""
    author_posts = [p for p in posts_db if p["author"].lower() == author_name.lower()]
    
    if not author_posts:
        return []
    
    return [BlogPostResponse(**post) for post in author_posts]

"""
===============================================================================
HOW TO RUN:
===============================================================================
uvicorn practice_01_blog_api:app --reload

===============================================================================
WHAT TO TEST:
===============================================================================

1. CREATE POSTS:
   - Create a valid post with all fields
   - Create a post without tags (should use default [])
   - Create unpublished post (should default to False)
   - Try invalid data:
     * Title too short (< 5 chars)
     * Content too short (< 50 chars)
     * Too many tags (> 5)

2. GET POSTS:
   - Get all posts
   - Get posts with pagination (?skip=2&limit=5)
   - Get specific post by ID
   - Try getting non-existent post (404 error)

3. UPDATE POSTS:
   - Use PUT to replace entire post
   - Use PATCH to update just title
   - Use PATCH to publish a post

4. DELETE POSTS:
   - Delete a post
   - Try deleting same post again (404 error)

5. SEARCH & FILTER:
   - Get only published posts
   - Search by tag
   - Get posts by author
   - Check stats

6. VALIDATION TESTS (should fail):
   POST /posts with:
   {
       "title": "Hi",  // Too short!
       "content": "Short",  // Too short!
       "author": "AB",  // Too short!
       "tags": ["a", "b", "c", "d", "e", "f"]  // Too many!
   }

===============================================================================
EXPECTED BEHAVIOR:
===============================================================================

✅ Tags are automatically converted to lowercase
✅ Duplicate tags are removed
✅ created_at is set automatically
✅ updated_at is set when post is modified
✅ post_id is auto-generated
✅ Validation prevents invalid data
✅ Clear error messages for invalid requests
✅ Proper HTTP status codes (201, 204, 404, etc.)

===============================================================================
SAMPLE TEST FLOW:
===============================================================================

1. Create 3 posts (2 published, 1 unpublished)
2. Get all posts
3. Search posts with tag "python"
4. Get only published posts
5. Update one post to publish it
6. Check stats
7. Delete one post
8. Verify it's gone

Go to http://localhost:8000/docs to test interactively!
"""