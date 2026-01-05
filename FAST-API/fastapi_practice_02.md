# Lesson 2 Practice Exercises

## 🎯 Exercise 1: Blog API (Beginner)

**Goal**: Build a simple blog system with posts and basic validation.

### Requirements:
Create a FastAPI app with the following:

**BlogPost Model** with these fields:
- `title` - string, 5-100 characters, required
- `content` - string, 50-5000 characters, required
- `author` - string, 3-50 characters, required
- `tags` - list of strings, optional, max 5 tags
- `published` - boolean, default False
- `created_at` - will be set automatically

**Endpoints to implement**:
1. `POST /posts` - Create a new blog post
2. `GET /posts` - Get all posts
3. `GET /posts/{post_id}` - Get specific post
4. `PUT /posts/{post_id}` - Update entire post
5. `DELETE /posts/{post_id}` - Delete a post
6. `GET /posts/published` - Get only published posts

**Bonus Challenges**:
- Add a custom validator to ensure tags are lowercase
- Add an endpoint to search posts by tag
- Add pagination (skip and limit) to GET /posts

### Success Criteria:
✅ All endpoints work correctly  
✅ Validation prevents invalid data  
✅ Can create, read, update, and delete posts  
✅ Try invalid data and see proper error messages  

---

## 🎯 Exercise 2: Student Management System (Intermediate)

**Goal**: Build a student enrollment system with nested models.

### Requirements:

**Models to create**:

**ContactInfo** (nested):
- `email` - valid email format
- `phone` - string, 10-15 characters
- `emergency_contact` - string

**Address** (nested):
- `street` - string
- `city` - string
- `zip_code` - string, exactly 5 or 6 digits

**Course** (for enrollment):
- `course_id` - integer
- `course_name` - string
- `credits` - integer, between 1-6

**Student** (main model):
- `student_id` - integer, will be auto-generated
- `first_name` - string, 2-50 characters
- `last_name` - string, 2-50 characters
- `age` - integer, between 16-100
- `contact` - ContactInfo (nested)
- `address` - Address (nested)
- `enrolled_courses` - list of Course objects
- `gpa` - float, between 0.0-4.0, optional

**Endpoints to implement**:
1. `POST /students` - Register new student
2. `GET /students` - Get all students
3. `GET /students/{student_id}` - Get specific student
4. `POST /students/{student_id}/enroll` - Enroll in a course
5. `GET /students/gpa` - Get students filtered by minimum GPA (query param)

**Bonus Challenges**:
- Add custom validator: total credits cannot exceed 18
- Add endpoint to calculate average GPA of all students
- Prevent duplicate course enrollment

### Success Criteria:
✅ Nested models validate correctly  
✅ Can register student with all nested data  
✅ Can enroll students in courses  
✅ GPA filtering works  
✅ Complex validation prevents invalid states  

---

## 🎯 Exercise 3: E-commerce Product Catalog (Advanced)

**Goal**: Build a complete product catalog with variants, inventory, and reviews.

### Requirements:

**ProductVariant** (nested):
- `variant_id` - integer
- `size` - enum: XS, S, M, L, XL
- `color` - string
- `sku` - string (unique identifier)
- `price` - float, must be positive
- `stock_quantity` - integer, 0 or greater

**Review** (nested):
- `reviewer_name` - string, 3-50 characters
- `rating` - integer, between 1-5
- `comment` - string, 10-500 characters
- `verified_purchase` - boolean
- `review_date` - datetime (auto-set)

**Product** (main model):
- `product_id` - auto-generated
- `name` - string, 3-100 characters
- `description` - string, 20-1000 characters
- `category` - enum: Electronics, Clothing, Books, Home, Sports
- `brand` - string, 2-50 characters
- `base_price` - float, must be positive
- `variants` - list of ProductVariant (at least 1 required)
- `reviews` - list of Review objects
- `is_active` - boolean, default True
- `tags` - list of strings, max 10 tags

**Endpoints to implement**:
1. `POST /products` - Add new product with variants
2. `GET /products` - List products with filters (category, min_price, max_price, in_stock)
3. `GET /products/{product_id}` - Get product details with all variants and reviews
4. `POST /products/{product_id}/reviews` - Add a review
5. `PATCH /products/{product_id}/variants/{variant_id}/stock` - Update variant stock
6. `GET /products/{product_id}/average-rating` - Calculate average rating

**Custom Validators to implement**:
- At least one variant must have stock > 0 for product to be active
- Variant SKU must be unique across all products
- Review rating average should be calculated from all reviews
- Tags must be unique (no duplicates)

**Bonus Challenges**:
- Add search endpoint (search by name/description)
- Add "related products" based on same category
- Track price history when variants are updated
- Add inventory alerts when stock is low

### Success Criteria:
✅ Complex nested structure works correctly  
✅ Multiple variants per product with individual stock  
✅ Review system with ratings  
✅ Advanced filtering and search  
✅ Custom validators enforce business logic  
✅ Can handle partial updates (PATCH)  

---

## 📝 Testing Checklist

For each exercise, test these scenarios:

### Valid Data Tests:
- ✅ Create with all required fields
- ✅ Create with optional fields included
- ✅ Create with optional fields omitted
- ✅ Update existing items
- ✅ Retrieve items correctly

### Invalid Data Tests (should return errors):
- ❌ Missing required fields
- ❌ Values outside allowed ranges (too short, too long, too big, too small)
- ❌ Wrong data types (string where number expected)
- ❌ Invalid email formats
- ❌ Violating custom validation rules
- ❌ Empty required lists
- ❌ Invalid enum values

---

## 🎓 Learning Tips

1. **Start Small**: Begin with Exercise 1, get it working, then move to 2
2. **Use the Docs**: Always test using `/docs` - it's interactive!
3. **Read Error Messages**: FastAPI gives great error details - read them!
4. **Test Invalid Data**: Learn by breaking things - try bad inputs
5. **Add One Feature at a Time**: Don't try to build everything at once

---

## 🆘 Stuck? Here are hints:

**For Exercise 1**:
```python
from pydantic import BaseModel, Field
from typing import List

class BlogPost(BaseModel):
    title: str = Field(..., min_length=5, max_length=100)
    # Add more fields...
```

**For Exercise 2**:
```python
class ContactInfo(BaseModel):
    email: EmailStr
    # Add more fields...

class Student(BaseModel):
    first_name: str
    contact: ContactInfo  # Nested!
    # Add more fields...
```

**For Exercise 3**:
```python
from enum import Enum

class Category(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    # Add more...

class ProductVariant(BaseModel):
    # Define variant fields...
    
    @validator('stock_quantity')
    def validate_stock(cls, v):
        if v < 0:
            raise ValueError('Stock cannot be negative')
        return v
```

---

## 🏆 When You're Done

After completing these exercises, you should be comfortable with:
- Creating Pydantic models with validation
- Nested models and lists of models
- Custom validators
- Request/response handling
- Error handling and status codes
- Building real-world API structures

**Ready for Lesson 3?** Once you complete at least Exercise 1 and 2, you'll be ready to learn about database integration!

---

## 💬 Need Help?

If you get stuck:
1. Check the example files (04, 05, 06) for similar patterns
2. Read the FastAPI docs: https://fastapi.tiangolo.com
3. Look at the Pydantic docs: https://docs.pydantic.dev
4. Ask me specific questions about what's not working!

Good luck! 🚀