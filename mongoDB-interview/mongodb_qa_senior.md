# MongoDB Senior Developer - Q&A Interview Guide

## Schema Design & Data Modeling

### Q1: When would you choose to embed documents versus referencing them? Give specific examples.

**Answer:**

The decision depends on access patterns, update frequency, and data size:

**Choose Embedding When:**

1. **One-to-Few Relationships** (1-10 items)
```javascript
// User with addresses
{
  _id: ObjectId("..."),
  name: "John Doe",
  addresses: [
    { type: "home", street: "123 Main St", city: "NYC" },
    { type: "work", street: "456 Park Ave", city: "NYC" }
  ]
}
```

2. **Data Read Together** (>80% of queries need both)
```javascript
// Blog post with comments
{
  _id: ObjectId("..."),
  title: "MongoDB Best Practices",
  content: "...",
  comments: [
    { author: "Alice", text: "Great post!", date: ISODate("...") },
    { author: "Bob", text: "Very helpful", date: ISODate("...") }
  ]
}
```

3. **Child Data Doesn't Change Independently**
```javascript
// Order with line items
{
  _id: ObjectId("..."),
  orderId: "ORD-123",
  items: [
    { product: "Laptop", price: 999, qty: 1 }  // Snapshot at purchase
  ]
}
```

**Choose Referencing When:**

1. **One-to-Many or Many-to-Many** (unbounded growth)
```javascript
// User with thousands of orders
{
  _id: ObjectId("user123"),
  name: "John Doe"
}

// Separate orders collection
{
  _id: ObjectId("..."),
  userId: ObjectId("user123"),
  total: 150.00
}
```

2. **Data Updated Frequently and Independently**
```javascript
// Product catalog - updated daily
{
  _id: ObjectId("prod123"),
  name: "iPhone",
  price: 999,  // Changes often
  stock: 50    // Changes often
}

// Orders reference products (don't embed)
{
  orderId: "ORD-456",
  items: [
    { productId: ObjectId("prod123"), quantity: 1 }
  ]
}
```

3. **Document Size Would Exceed 16MB Limit**

4. **Data Needs to be Accessed Independently**
```javascript
// Users and Organizations (many-to-many)
// Keep separate, reference by ID
```

**Rule of Thumb:**
- Embed for "contains" relationships (Order contains Items)
- Reference for "has" relationships (User has Orders)

---

### Q2: Explain the difference between normalization and denormalization in MongoDB. When is each appropriate?

**Answer:**

**Normalization** = Breaking data into separate collections (like SQL)
**Denormalization** = Duplicating data across documents

**Normalized Approach:**
```javascript
// Users collection
{ _id: ObjectId("user1"), name: "Alice", email: "alice@example.com" }

// Posts collection
{ 
  _id: ObjectId("post1"), 
  authorId: ObjectId("user1"),  // Reference only
  title: "My First Post" 
}

// To display post with author: Need $lookup (JOIN)
db.posts.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "authorId",
      foreignField: "_id",
      as: "author"
    }
  }
])
```

**Denormalized Approach:**
```javascript
// Posts collection with embedded author info
{ 
  _id: ObjectId("post1"),
  title: "My First Post",
  author: {
    id: ObjectId("user1"),
    name: "Alice",        // Duplicated data
    email: "alice@example.com"  // Duplicated data
  }
}

// No $lookup needed - single query!
db.posts.find({ _id: ObjectId("post1") })
```

**When to Normalize:**
✅ Data changes frequently (product prices, user profiles)
✅ Data is large and rarely accessed together
✅ Need strong consistency across documents
✅ Many-to-many relationships

**When to Denormalize:**
✅ Read-heavy workload (90%+ reads)
✅ Data changes infrequently (author name, historical prices)
✅ Need fast queries (avoid JOINs)
✅ Accept eventual consistency

**Hybrid Approach (Recommended):**
```javascript
// Store both reference AND frequently used fields
{
  _id: ObjectId("post1"),
  title: "My First Post",
  authorId: ObjectId("user1"),      // Reference for updates
  authorName: "Alice",               // Denormalized for display
  authorAvatar: "url.jpg"            // Denormalized for display
}
```

**Update Strategy for Denormalized Data:**
```javascript
// When user changes name, update all posts
db.posts.updateMany(
  { authorId: ObjectId("user1") },
  { $set: { authorName: "Alice Smith" } }
)
```

---

## Indexing & Performance

### Q3: What is a compound index? How does field order affect performance?

**Answer:**

A **compound index** indexes multiple fields in a specific order.

**Example:**
```javascript
db.orders.createIndex({ status: 1, createdAt: -1, total: 1 })
```

**Field Order Matters - ESR Rule:**

**E**quality → **S**ort → **R**ange

1. **Equality fields first** (exact matches)
2. **Sort fields next**
3. **Range fields last** (>, <, $in)

**Example Query:**
```javascript
db.orders.find({
  status: "pending",              // Equality
  total: { $gt: 100 }             // Range
}).sort({ createdAt: -1 })        // Sort
```

**Good Index (follows ESR):**
```javascript
db.orders.createIndex({ 
  status: 1,      // E - Equality
  createdAt: -1,  // S - Sort
  total: 1        // R - Range
})
```

**Bad Index (doesn't follow ESR):**
```javascript
db.orders.createIndex({ 
  total: 1,       // Range first - BAD!
  status: 1,
  createdAt: -1
})
// Less efficient - can't use index effectively
```

**How Index is Used:**

```javascript
// Index: { status: 1, createdAt: -1, total: 1 }

// Query 1: Uses full index
db.orders.find({ status: "pending", total: { $gt: 100 } })
         .sort({ createdAt: -1 })

// Query 2: Uses partial index (status only)
db.orders.find({ status: "pending" })

// Query 3: Can't use index efficiently
db.orders.find({ total: { $gt: 100 } })
// Needs status to use this index

// Query 4: Uses index for sorting too
db.orders.find({ status: "pending" }).sort({ createdAt: -1 })
```

**Index Prefixes:**
```javascript
// Index: { a: 1, b: 1, c: 1 }

// These can use the index:
{ a: 1 }              ✅ Prefix: a
{ a: 1, b: 1 }        ✅ Prefix: a, b
{ a: 1, b: 1, c: 1 }  ✅ Full index

// These can't use the index efficiently:
{ b: 1 }              ❌ Missing 'a'
{ c: 1 }              ❌ Missing 'a' and 'b'
{ b: 1, c: 1 }        ❌ Missing 'a'
```

**Practical Example:**
```javascript
// E-commerce orders
db.orders.createIndex({ 
  userId: 1,          // E - Filter by user
  status: 1,          // E - Filter by status
  createdAt: -1       // S - Sort by date (newest first)
})

// This query is optimized:
db.orders.find({ 
  userId: ObjectId("user123"), 
  status: "completed" 
}).sort({ createdAt: -1 })

// Returns: IXSCAN (uses index efficiently)
```

**Verify with explain():**
```javascript
db.orders.find({ status: "pending" })
  .sort({ createdAt: -1 })
  .explain("executionStats")

// Look for:
// - "stage": "IXSCAN" (good - uses index)
// - "stage": "COLLSCAN" (bad - scans all documents)
// - "executionStats.totalDocsExamined" should be close to "nReturned"
```

---

### Q4: What are covered queries and why are they important?

**Answer:**

A **covered query** is one where all requested fields are in the index, so MongoDB doesn't need to access the actual documents.

**Normal Query (Not Covered):**
```javascript
// Index: { status: 1 }
db.orders.find(
  { status: "pending" },
  { _id: 1, status: 1, total: 1 }  // Need total from document
)

// Process:
// 1. Use index to find matching documents
// 2. Read documents from disk to get 'total' field
// 3. Return results
```

**Covered Query:**
```javascript
// Index: { status: 1, total: 1 }
db.orders.find(
  { status: "pending" },
  { _id: 0, status: 1, total: 1 }  // All fields in index!
)

// Process:
// 1. Use index to find matches
// 2. Return data directly from index (no disk read!)
// 3. Much faster!
```

**Key Requirements for Covered Queries:**

1. ✅ All queried fields must be in index
2. ✅ All returned fields must be in index
3. ✅ Must exclude `_id` (unless it's in index)
4. ✅ No array fields in index
5. ✅ Query must not be on embedded documents

**Example - User Search:**
```javascript
// Create covering index
db.users.createIndex({ 
  email: 1, 
  name: 1, 
  status: 1 
})

// Covered query (fast!)
db.users.find(
  { email: "john@example.com" },
  { _id: 0, email: 1, name: 1, status: 1 }  // Exclude _id!
)

// Verify it's covered
.explain("executionStats")

// Look for:
// "totalDocsExamined": 0  ← Covered! (no documents read)
// "totalKeysExamined": 1  ← Only index used
```

**Why Important:**

✅ **10x-100x faster** - No disk I/O for documents
✅ **Less memory usage** - Index pages stay in cache
✅ **Better for high-traffic queries** - Scalability

**Common Mistakes:**
```javascript
// ❌ Including _id prevents covering
db.users.find(
  { email: "john@example.com" },
  { email: 1, name: 1 }  // _id is included by default!
)

// ✅ Explicitly exclude _id
db.users.find(
  { email: "john@example.com" },
  { _id: 0, email: 1, name: 1 }
)

// ❌ Querying fields not in index
db.users.find(
  { email: "john@example.com" },
  { _id: 0, email: 1, name: 1, address: 1 }  // address not in index
)
```

---

### Q5: How do you identify and fix slow queries in MongoDB?

**Answer:**

**Step 1: Enable Profiling**
```javascript
// Profile slow queries (>100ms)
db.setProfilingLevel(1, { slowms: 100 })

// Profile all queries (development only!)
db.setProfilingLevel(2)

// Check profiling status
db.getProfilingStatus()
```

**Step 2: Review Slow Queries**
```javascript
// Find slowest queries
db.system.profile.find({
  millis: { $gt: 100 }
}).sort({ millis: -1 }).limit(10).pretty()

// Aggregate by operation type
db.system.profile.aggregate([
  { $match: { millis: { $gt: 100 } } },
  { $group: {
    _id: "$ns",  // Collection name
    count: { $sum: 1 },
    avgMs: { $avg: "$millis" },
    maxMs: { $max: "$millis" }
  }},
  { $sort: { count: -1 } }
])
```

**Step 3: Analyze with explain()**
```javascript
// The slow query
db.orders.find({ 
  status: "pending", 
  total: { $gt: 100 } 
}).explain("executionStats")

// Key metrics to check:
{
  "executionStats": {
    "executionTimeMillis": 523,      // Total time
    "totalDocsExamined": 100000,     // Documents scanned
    "totalKeysExamined": 100000,     // Index entries scanned
    "nReturned": 50,                 // Documents returned
    
    "executionStages": {
      "stage": "COLLSCAN"            // ❌ BAD - full collection scan
      // Should be "IXSCAN" ✅
    }
  }
}
```

**Step 4: Common Issues & Fixes**

**Issue 1: Missing Index (COLLSCAN)**
```javascript
// Problem
db.orders.find({ status: "pending" })
// Stage: COLLSCAN (scans all documents)

// Fix: Add index
db.orders.createIndex({ status: 1 })
// Stage: IXSCAN (uses index)
```

**Issue 2: Inefficient Index**
```javascript
// Problem: Using only part of index
db.orders.find({ 
  total: { $gt: 100 }  // Range query
}).sort({ createdAt: -1 })

// Current index: { status: 1, total: 1 }
// Doesn't help without status!

// Fix: Create appropriate index
db.orders.createIndex({ total: 1, createdAt: -1 })
```

**Issue 3: Examining Too Many Documents**
```javascript
// Problem
{
  "totalDocsExamined": 10000,
  "nReturned": 10
}
// Examined 10,000 to return 10!

// Fix: Add selective index
db.orders.createIndex({ 
  status: 1,      // High selectivity
  customerId: 1 
})
```

**Issue 4: Large Documents**
```javascript
// Problem: Returning 1MB documents
db.products.find({ category: "Electronics" })

// Fix: Use projection
db.products.find(
  { category: "Electronics" },
  { name: 1, price: 1, _id: 0 }  // Only needed fields
)
```

**Issue 5: Regex Without Index**
```javascript
// Problem: Slow regex
db.users.find({ email: /.*@example.com/ })
// Can't use index efficiently

// Fix 1: Anchor regex to start
db.users.find({ email: /^john.*@example.com/ })
// Can use index

// Fix 2: Use text index
db.users.createIndex({ email: "text" })
db.users.find({ $text: { $search: "example.com" } })
```

**Issue 6: Unoptimized Aggregation**
```javascript
// Problem: $lookup before $match
db.orders.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  },
  { $match: { status: "pending" } }  // Match after lookup!
])

// Fix: Match first to reduce documents
db.orders.aggregate([
  { $match: { status: "pending" } },  // Filter early ✅
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  }
])
```

**Step 5: Monitoring & Alerts**
```javascript
// Set up regular monitoring
function checkSlowQueries() {
  const slowQueries = db.system.profile.find({
    ts: { $gt: new Date(Date.now() - 3600000) },  // Last hour
    millis: { $gt: 1000 }  // Slower than 1 second
  }).count()
  
  if (slowQueries > 10) {
    alertTeam('Too many slow queries: ' + slowQueries)
  }
}

// Run every hour
setInterval(checkSlowQueries, 3600000)
```

**Quick Checklist:**
- ✅ All queries have appropriate indexes
- ✅ Indexes follow ESR rule
- ✅ Use projections to limit returned data
- ✅ Filter early in aggregation pipelines
- ✅ Monitor with explain() regularly
- ✅ Review system.profile periodically

---

## Aggregation Framework

### Q6: Explain the difference between $lookup and embedding. When would you use each?

**Answer:**

**$lookup** = JOIN in MongoDB (combines data from multiple collections)
**Embedding** = Store related data in same document

**$lookup Example:**
```javascript
// Orders collection
{ _id: 1, userId: ObjectId("user123"), total: 150 }

// Users collection  
{ _id: ObjectId("user123"), name: "John", email: "john@example.com" }

// Use $lookup to join
db.orders.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "userDetails"
    }
  }
])

// Result:
{
  _id: 1,
  userId: ObjectId("user123"),
  total: 150,
  userDetails: [
    { _id: ObjectId("user123"), name: "John", email: "john@example.com" }
  ]
}
```

**Embedding Example:**
```javascript
// Everything in one document
{
  _id: 1,
  total: 150,
  user: {
    userId: ObjectId("user123"),
    name: "John",
    email: "john@example.com"
  }
}

// No $lookup needed - single query!
db.orders.find({ _id: 1 })
```

**When to Use $lookup:**

✅ **Data Changes Frequently**
```javascript
// User profile updated often
// Don't embed in every order - use $lookup
db.orders.aggregate([
  { $match: { _id: 1 } },
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "currentUserInfo"  // Always up-to-date
    }
  }
])
```

✅ **Many-to-Many Relationships**
```javascript
// Products can be in many orders
// Orders can have many products
// Keep separate, use $lookup when needed
```

✅ **Large Arrays (would exceed 16MB)**
```javascript
// User with 100,000 orders
// Store separately, use $lookup with pagination
db.orders.find({ userId: userId }).limit(20).skip(page * 20)
```

✅ **Data Accessed Independently**
```javascript
// Analytics on users without loading orders
db.users.find({ registeredDate: { $gte: today } })
```

**When to Use Embedding:**

✅ **Data Read Together (>80% of queries)**
```javascript
// Blog post with comments - always displayed together
{
  _id: 1,
  title: "My Post",
  content: "...",
  comments: [
    { author: "Alice", text: "Great!" },
    { author: "Bob", text: "Helpful" }
  ]
}
```

✅ **One-to-Few Relationships**
```javascript
// Person with 2-3 addresses
{
  _id: 1,
  name: "John",
  addresses: [
    { type: "home", street: "123 Main St" },
    { type: "work", street: "456 Park Ave" }
  ]
}
```

✅ **Historical Data (snapshot)**
```javascript
// Order with product prices at purchase time
{
  _id: 1,
  items: [
    { 
      productId: ObjectId("..."),
      name: "Laptop",
      price: 999  // Price when ordered (don't change)
    }
  ]
}
```

✅ **Performance Critical**
```javascript
// Real-time dashboard - can't afford $lookup delay
// Embed critical metrics
{
  userId: 1,
  stats: {
    totalOrders: 45,
    totalSpent: 5600,
    lastOrderDate: ISODate("...")
  }
}
```

**Performance Comparison:**

```javascript
// Embedded (1 query)
db.orders.find({ _id: 1 })
// ~5ms

// $lookup (slower)
db.orders.aggregate([
  { $match: { _id: 1 } },
  { $lookup: { from: "users", ... } }
])
// ~50ms (10x slower!)
```

**Hybrid Approach (Best Practice):**
```javascript
// Store frequently used fields + reference
{
  _id: 1,
  userId: ObjectId("user123"),     // Reference for updates
  userName: "John",                 // Cached for display
  userEmail: "john@example.com",   // Cached for display
  total: 150
}

// Update cached fields when user changes
db.orders.updateMany(
  { userId: ObjectId("user123") },
  { $set: { userName: "John Smith" } }
)

// Use $lookup only when need full user details
```

---

### Q7: How do you optimize aggregation pipelines for better performance?

**Answer:**

**Principle: Filter Early, Minimize Data Movement**

**1. Put $match at the Beginning**
```javascript
// ❌ Bad: Processes all documents
db.orders.aggregate([
  { $unwind: "$items" },
  { $lookup: { from: "products", ... } },
  { $match: { status: "completed" } }  // Too late!
])

// ✅ Good: Filter first
db.orders.aggregate([
  { $match: { status: "completed" } },  // Reduces 90% of data
  { $unwind: "$items" },
  { $lookup: { from: "products", ... } }
])
```

**2. Use $project to Remove Unnecessary Fields**
```javascript
// ❌ Bad: Carries large documents through pipeline
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: { 
    _id: "$customerId",
    total: { $sum: "$amount" }
  }}
  // Large order documents processed unnecessarily
])

// ✅ Good: Project early
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $project: { customerId: 1, amount: 1 } },  // Only needed fields
  { $group: { 
    _id: "$customerId",
    total: { $sum: "$amount" }
  }}
])
```

**3. Optimize $lookup with Pipeline**
```javascript
// ❌ Bad: Joins all user documents
db.orders.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  }
])

// ✅ Good: Filter in subpipeline
db.orders.aggregate([
  {
    $lookup: {
      from: "users",
      let: { userId: "$userId" },
      pipeline: [
        { $match: { 
          $expr: { $eq: ["$_id", "$$userId"] },
          status: "active"  // Filter here!
        }},
        { $project: { name: 1, email: 1 } }  // Only needed fields
      ],
      as: "user"
    }
  }
])
```

**4. Use Indexes in $match and $sort**
```javascript
// Create index first
db.orders.createIndex({ status: 1, orderDate: -1 })

// Pipeline uses index for $match and $sort
db.orders.aggregate([
  { $match: { status: "completed" } },  // Uses index
  { $sort: { orderDate: -1 } },         // Uses index
  { $limit: 100 }
])

// Verify with explain
.explain("executionStats")
// Look for "IXSCAN"
```

**5. Limit Results Early**
```javascript
// ✅ Good: Limit early if possible
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $sort: { total: -1 } },
  { $limit: 10 },  // Stop processing after 10
  { $lookup: { ... } }  // Only for 10 documents
])
```

**6. Use $facet Wisely**
```javascript
// Process multiple aggregations in one pass
db.orders.aggregate([
  { $match: { orderDate: { $gte: startDate } } },
  {
    $facet: {
      // Count total
      totalCount: [
        { $count: "count" }
      ],
      // Get top products
      topProducts: [
        { $unwind: "$items" },
        { $group: { _id: "$items.productId", total: { $sum: "$items.quantity" } } },
        { $sort: { total: -1 } },
        { $limit: 10 }
      ],
      // Calculate revenue
      revenue: [
        { $group: { _id: null, total: { $sum: "$total" } } }
      ]
    }
  }
])
// Processes data once for all three metrics
```

**7. Avoid $lookup in Loops**
```javascript
// ❌ Very Bad: $lookup inside $map
db.orders.aggregate([
  {
    $addFields: {
      enrichedItems: {
        $map: {
          input: "$items",
          as: "item",
          in: {
            // This runs $lookup for EACH item!
            // Extremely slow!
          }
        }
      }
    }
  }
])

// ✅ Good: Single $lookup with $unwind
db.orders.aggregate([
  { $unwind: "$items" },
  {
    $lookup: {
      from: "products",
      localField: "items.productId",
      foreignField: "_id",
      as: "productInfo"
    }
  },
  { $unwind: "$productInfo" },
  {
    $group: {
      _id: "$_id",
      items: { 
        $push: { 
          item: "$items", 
          product: "$productInfo" 
        } 
      }
    }
  }
])
```

**8. Use allowDiskUse for Large Data**
```javascript
// For aggregations that exceed 100MB memory limit
db.orders.aggregate([
  { $match: { ... } },
  { $sort: { total: -1 } },
  { $group: { ... } }
], { allowDiskUse: true })

// But try to avoid by optimizing first!
```

**9. Monitor Performance**
```javascript
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: { _id: "$customerId", total: { $sum: "$total" } } }
], { explain: true })

// Check executionStats:
// - totalDocsExamined (should be low)
// - executionTimeMillis (should be fast)
// - usedDisk: false (should fit in memory)
```

**Real Example - Before/After:**

❌ **Before (Slow - 5 seconds):**
```javascript
db.orders.aggregate([
  { $unwind: "$items" },
  {
    $lookup: {
      from: "products",
      localField: "items.productId",
      foreignField: "_id",
      as: "product"
    }
  },
  { $unwind: "$product" },
  { $match: { 
    "product.category": "Electronics",
    status: "completed"
  }},
  {
    $group: {
      _id: "$product._id",
      totalSold: { $sum: "$items.quantity" }
    }
  },
  { $sort: { totalSold: -1 } },
  { $limit: 10 }
])
```

✅ **After (Fast - 200ms):**
```javascript
db.orders.aggregate([
  // Filter first
  { $match: { status: "completed" } },
  { $unwind: "$items" },
  
  // Optimized lookup
  {
    $lookup: {
      from: "products",
      let: { productId: "$items.productId" },
      pipeline: [
        { $match: { 
          $expr: { $eq: ["$_id", "$$productId"] },
          category: "Electronics"  // Filter in subpipeline
        }},
        { $project: { _id: 1, name: 1 } }  // Only needed fields
      ],
      as: "product"
    }
  },
  { $unwind: "$product" },
  
  // Group
  {
    $group: {
      _id: "$product._id",
      name: { $first: "$product.name" },
      totalSold: { $sum: "$items.quantity" }
    }
  },
  
  // Sort and limit
  { $sort: { totalSold: -1 } },
  { $limit: 10 }
])
```

**Performance Checklist:**
- ✅ $match as early as possible
- ✅ Use indexes for $match and $sort
- ✅ $project to remove unneeded fields
- ✅ $limit early when possible
- ✅ Optimize $lookup with pipelines
- ✅ Avoid $lookup in loops
- ✅ Monitor with explain()

---

## Replication & High Availability

### Q8: Explain MongoDB replica sets. How do they ensure high availability?

**Answer:**

A **Replica Set** is a group of MongoDB servers that maintain the same data for redundancy and high availability.

**Architecture:**
```
Primary (Read/Write)
    ↓ replicates to
Secondary (Read only) ← Can be promoted to Primary
    ↓ replicates to  
Secondary (Read only) ← Can be promoted to Primary
```

**How It Works:**

**1. Normal Operation:**
```javascript
// Application writes to PRIMARY
const result = await db.orders.insertOne({
  orderId: "ORD-123",
  total: 150
})

// PRIMARY writes to oplog (operation log)
// SECONDARIES read oplog and apply operations
// All servers eventually have same data
```

**2. Automatic Failover:**
```
Time 0: PRIMARY is healthy
    Primary: Server A ✅
    Secondary: Server B
    Secondary: Server C

Time 1: PRIMARY crashes
    Primary: Server A ❌ (crashed)
    Secondary: Server B (detects failure)
    Secondary: Server C (detects failure)

Time 2: Election happens (takes 10-12 seconds)
    Server B: Votes for self
    Server C: Votes for B
    ➜ Server B becomes new PRIMARY ✅

Time 3: Normal operation restored
    Primary: Server B ✅ (promoted)
    Secondary: Server C
    Former Primary: Server A (when it comes back, becomes Secondary)
```

**3. Read Preferences:**
```javascript
// Default: Read from PRIMARY only (strongest consistency)
db.orders.find({}).readPref("primary")

// Read from SECONDARY (reduce primary load, eventual consistency)
db.orders.find({}).readPref("secondary")

// Read from PRIMARY preferred, fallback to SECONDARY
db.orders.find({}).readPref("primaryPreferred")

// Read from SECONDARY preferred, fallback to PRIMARY  
db.orders.find({}).readPref("secondaryPreferred")

// Read from nearest (lowest latency)
db.orders.find({}).readPref("nearest")
```

**4. Write Concerns:**
```javascript
// Wait for write to be acknowledged by PRIMARY only (fast, risky)
await db.orders.insertOne(doc, { writeConcern: { w: 1 } })

// Wait for MAJORITY (at least 2 of 3 servers) - recommended
await db.orders.insertOne(doc, { writeConcern: { w: "majority" } })

// Wait for ALL servers (slow but most durable)
await db.orders.insertOne(doc, { writeConcern: { w: 3 } })

// Write with journal (persisted to disk)
await db.orders.insertOne(doc, { 
  writeConcern: { w: "majority", j: true } 
})
```

**5. Replica Set Configuration:**
```javascript
// Initialize replica set
rs.initiate({
  _id: "myReplicaSet",
  members: [
    { _id: 0, host: "server1:27017", priority: 2 },  // Preferred primary
    { _id: 1, host: "server2:27017", priority: 1 },
    { _id: 2, host: "server3:27017", priority: 0, hidden: true }  // For backups
  ]
})

// Check status
rs.status()

// Add member
rs.add("server4:27017")

// Remove member
rs.remove("server4:27017")
```

**6. Common Configurations:**

**3-Member Replica Set (Most Common):**
```
Primary: server1 (writes)
Secondary: server2 (replica)
Secondary: server3 (replica)

Benefits:
- Survives 1 server failure
- Can do maintenance on 1 server
- Recommended for production
```

**5-Member Replica Set (High Availability):**
```
Primary: server1
Secondary: server2
Secondary: server3
Secondary: server4
Arbiter: server5 (voting only, no data)

Benefits:
- Survives 2 server failures
- Higher availability
- Use for critical systems
```

**Geographic Distribution:**
```
Data Center 1:
  - Primary
  - Secondary
  
Data Center 2:
  - Secondary
  - Arbiter

Benefits:
- Survives datacenter failure
- Lower latency for distributed users
```

**7. How High Availability is Ensured:**

✅ **Automatic Failover**: New primary elected in 10-12 seconds
✅ **Data Redundancy**: Multiple copies of data
✅ **No Single Point of Failure**: Any member can become primary
✅ **Rolling Maintenance**: Upgrade one server at a time
✅ **Read Scaling**: Distribute reads across secondaries

**8. Monitoring:**
```javascript
// Check replica lag
rs.printReplicationInfo()

// Check member health
rs.status()

// OpLog size
db.getReplicationInfo()
```

---

### Q9: What is sharding in MongoDB? When would you implement it?

**Answer:**

**Sharding** = Horizontal scaling by distributing data across multiple servers (shards)

**Architecture:**
```
Application
    ↓
Mongos (Router) - Directs queries to correct shard
    ↓
Config Servers - Store metadata about shards
    ↓
Shard 1 (Replica Set) - Users A-M
Shard 2 (Replica Set) - Users N-Z
Shard 3 (Replica Set) - Products
```

**When to Shard:**

✅ **Data Size > Server RAM**
```
Single server: 64GB RAM
Data size: 500GB
➜ Working set doesn't fit in memory
➜ Need sharding to distribute
```

✅ **Write Throughput Exceeds Single Server**
```
Single server: 10K writes/sec max
Your app: 50K writes/sec
➜ Need 5+ shards to distribute writes
```

✅ **Geographic Distribution**
```
Users in US, Europe, Asia
➜ Shard by region for low latency
```

✅ **Storage Limits**
```
Single server: 2TB disk
Your data: 10TB
➜ Distribute across multiple servers
```

**When NOT to Shard:**

❌ Data < 100GB (use replica sets instead)
❌ Low traffic (< 1000 QPS)
❌ Can scale vertically (add RAM/CPU)
❌ Queries don't include shard key (scatter-gather queries)

**Shard Key Selection (Most Critical Decision):**

**Option 1: Hashed Shard Key**
```javascript
sh.shardCollection("mydb.users", { _id: "hashed" })

Pros:
✅ Even distribution
✅ Good for random inserts

Cons:
❌ Range queries hit all shards
❌ Can't use for sorting efficiently
```

**Option 2: Range-Based Shard Key**
```javascript
sh.shardCollection("mydb.users", { userId: 1 })

// Data distributed by ranges:
// Shard 1: userId 1-1000000
// Shard 2: userId 1000001-2000000
// Shard 3: userId 2000001-3000000

Pros:
✅ Range queries hit fewer shards
✅ Can sort efficiently

Cons:
❌ May have hot spots (uneven distribution)
❌ Sequential inserts go to same shard
```

**Option 3: Compound Shard Key (Best for Most Cases)**
```javascript
sh.shardCollection("mydb.orders", { customerId: 1, orderDate: 1 })

Pros:
✅ Better distribution
✅ Targets queries by customer
✅ Can range on date

Cons:
❌ More complex to manage
```

**Good Shard Key Characteristics:**

✅ **High Cardinality** (many unique values)
```javascript
// Good
{ customerId: 1 }  // Millions of customers

// Bad
{ status: 1 }  // Only 5 values (pending, completed, etc.)
```

✅ **Even Distribution**
```javascript
// Good
{ userId: "hashed" }  // Evenly distributed

// Bad
{ country: 1 }  // 80% users in US, 20% rest
```

✅ **Query Pattern Aligned**
```javascript
// If most queries are: find({ customerId: X })
// Good shard key: { customerId: 1 }

// If most queries are: find({ status: "pending" })
// Bad shard key: { customerId: 1 } (scatter-gather)
```

**Example Implementation:**

**1. Enable Sharding:**
```javascript
// Connect to mongos
mongosh --host mongos.example.com

// Enable sharding on database
sh.enableSharding("ecommerce")

// Shard orders collection
sh.shardCollection("ecommerce.orders", { customerId: 1, orderDate: 1 })
```

**2. Zone Sharding (Geographic):**
```javascript
// Add shards to zones
sh.addShardToZone("shard0", "US")
sh.addShardToZone("shard1", "EU")
sh.addShardToZone("shard2", "ASIA")

// Define zone ranges
sh.updateZoneKeyRange(
  "ecommerce.users",
  { region: "US", userId: MinKey },
  { region: "US", userId: MaxKey },
  "US"
)

sh.updateZoneKeyRange(
  "ecommerce.users",
  { region: "EU", userId: MinKey },
  { region: "EU", userId: MaxKey },
  "EU"
)

// Data automatically routed to correct region
```

**3. Monitoring:**
```javascript
// Check shard distribution
sh.status()

// Check if collection is sharded
db.orders.getShardDistribution()

// Example output:
// Shard shard0: 33% (330000 docs)
// Shard shard1: 34% (340000 docs)
// Shard shard2: 33% (330000 docs)
```

**Common Issues:**

**Issue 1: Jumbo Chunks**
```javascript
// Chunk grows > 64MB, can't split
// Caused by low cardinality shard key

// Check for jumbo chunks
sh.status()

// Fix: Choose better shard key with higher cardinality
```

**Issue 2: Hot Shards**
```javascript
// One shard getting all writes
// Caused by monotonically increasing shard key (like timestamp)

// Bad: All new orders go to last shard
sh.shardCollection("orders", { orderDate: 1 })

// Good: Distribute evenly
sh.shardCollection("orders", { customerId: "hashed", orderDate: 1 })
```

**Issue 3: Scatter-Gather Queries**
```javascript
// Query without shard key hits ALL shards (slow)
db.orders.find({ status: "pending" })
// Hits all 10 shards

// Include shard key to target specific shard
db.orders.find({ customerId: 12345, status: "pending" })
// Hits only 1 shard
```

**Best Practices:**

✅ Test shard key thoroughly before production
✅ Monitor chunk distribution regularly
✅ Include shard key in all queries when possible
✅ Plan for data growth (3-5 years)
✅ Start with replica sets, add sharding when needed
✅ Use compound shard keys for flexibility
✅ Consider using hashed component for even distribution

---

## Transactions & Consistency

### Q10: When should you use multi-document transactions in MongoDB? What are the trade-offs?

**Answer:**

**Multi-document transactions** ensure ACID properties across multiple documents/collections.

**When to Use:**

✅ **Financial Operations**
```javascript
// Transfer money between accounts
const session = client.startSession();
await session.withTransaction(async () => {
  // Deduct from account A
  await db.accounts.updateOne(
    { accountId: "A", balance: { $gte: 100 } },
    { $inc: { balance: -100 } },
    { session }
  );
  
  // Add to account B
  await db.accounts.updateOne(
    { accountId: "B" },
    { $inc: { balance: 100 } },
    { session }
  );
});
// Both succeed or both fail
```

✅ **Inventory Management**
```javascript
await session.withTransaction(async () => {
  // Reserve inventory
  const result = await db.products.updateOne(
    { _id: productId, stock: { $gte: quantity } },
    { $inc: { stock: -quantity } },
    { session }
  );
  
  if (result.modifiedCount === 0) {
    throw new Error("Insufficient stock");
  }
  
  // Create order
  await db.orders.insertOne({ ... }, { session });
  
  // Update user stats
  await db.users.updateOne(
    { _id: userId },
    { $inc: { totalOrders: 1 } },
    { session }
  );
});
```

✅ **Complex Multi-Step Operations**
```javascript
await session.withTransaction(async () => {
  // Step 1: Create user
  const user = await db.users.insertOne({ ... }, { session });
  
  // Step 2: Create user profile
  await db.profiles.insertOne({ userId: user.insertedId, ... }, { session });
  
  // Step 3: Add to organization
  await db.organizations.updateOne(
    { _id: orgId },
    { $push: { members: user.insertedId } },
    { session }
  );
  
  // Step 4: Send notification
  await db.notifications.insertOne({ userId: user.insertedId, ... }, { session });
});
```

**When NOT to Use:**

❌ **Single Document Operations** (Already atomic)
```javascript
// Don't need transaction - already atomic
db.users.updateOne(
  { _id: userId },
  { 
    $inc: { balance: -100 },
    $push: { transactions: { amount: 100, date: new Date() } }
  }
)
```

❌ **Performance-Critical Paths** (Transactions are slower)
```javascript
// High-throughput logging - don't use transactions
db.logs.insertOne({ level: "info", message: "..." })
```

❌ **Long-Running Operations**
```javascript
// Bad - transaction holds locks too long
await session.withTransaction(async () => {
  const data = await fetchFromExternalAPI();  // Slow!
  await processLargeFile(data);  // Slow!
  await db.collection.insertOne(data, { session });
});

// Good - minimize transaction scope
const data = await fetchFromExternalAPI();
const processed = await processLargeFile(data);

await session.withTransaction(async () => {
  await db.collection.insertOne(processed, { session });
});
```

**Trade-offs:**

**Performance Impact:**
```javascript
// Without transaction: ~5ms
await db.orders.insertOne({ ... });

// With transaction: ~20ms (4x slower)
await session.withTransaction(async () => {
  await db.orders.insertOne({ ... }, { session });
});
```

**Resource Usage:**
- ✅ Transactions use more memory
- ✅ Hold locks longer
- ✅ Increase latency
- ✅ Limited to 60 seconds by default

**Best Practices:**

**1. Keep Transactions Short**
```javascript
// ✅ Good
await session.withTransaction(async () => {
  await db.collection1.updateOne({ ... }, { session });
  await db.collection2.updateOne({ ... }, { session });
});

// ❌ Bad
await session.withTransaction(async () => {
  await sleep(1000);  // Don't do this!
  await callExternalAPI();  // Don't do this!
  await db.collection1.updateOne({ ... }, { session });
});
```

**2. Handle Errors Properly**
```javascript
const session = client.startSession();
try {
  await session.withTransaction(async () => {
    // Transaction logic
    
    if (someCondition) {
      throw new Error("Business rule violation");
      // Automatic rollback
    }
  });
} catch (error) {
  console.error("Transaction failed:", error);
  // Already rolled back automatically
} finally {
  await session.endSession();
}
```

**3. Retry on Transient Errors**
```javascript
async function executeWithRetry(operation, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (error.hasErrorLabel('TransientTransactionError') && 
          attempt < maxRetries - 1) {
        await sleep(100 * Math.pow(2, attempt));  // Exponential backoff
        continue;
      }
      throw error;
    }
  }
}

await executeWithRetry(async () => {
  const session = client.startSession();
  try {
    return await session.withTransaction(async () => {
      // Transaction logic
    });
  } finally {
    await session.endSession();
  }
});
```

**4. Use Write Concerns**
```javascript
await session.withTransaction(
  async () => {
    // Transaction logic
  },
  {
    readPreference: 'primary',
    readConcern: { level: 'snapshot' },
    writeConcern: { w: 'majority', wtimeout: 5000 }
  }
);
```

**Alternatives to Transactions:**

**1. Two-Phase Commit Pattern**
```javascript
// For operations that don't need immediate consistency
async function transferWithTwoPhase(fromAccount, toAccount, amount) {
  const txnId = new ObjectId();
  
  // Phase 1: Prepare
  await db.transactions.insertOne({
    _id: txnId,
    state: "pending",
    from: fromAccount,
    to: toAccount,
    amount: amount
  });
  
  await db.accounts.updateOne(
    { _id: fromAccount },
    { $inc: { pendingDebits: amount } }
  );
  
  // Phase 2: Commit
  try {
    await db.accounts.updateOne(
      { _id: fromAccount },
      { 
        $inc: { balance: -amount, pendingDebits: -amount }
      }
    );
    
    await db.accounts.updateOne(
      { _id: toAccount },
      { $inc: { balance: amount } }
    );
    
    await db.transactions.updateOne(
      { _id: txnId },
      { $set: { state: "committed" } }
    );
  } catch (error) {
    // Rollback
    await db.accounts.updateOne(
      { _id: fromAccount },
      { $inc: { pendingDebits: -amount } }
    );
    
    await db.transactions.updateOne(
      { _id: txnId },
      { $set: { state: "cancelled" } }
    );
  }
}
```

**2. Eventual Consistency with Change Streams**
```javascript
// Update one collection, listen for changes, update others
const changeStream = db.orders.watch();

changeStream.on('change', async (change) => {
  if (change.operationType === 'insert') {
    const order = change.fullDocument;
    
    // Asynchronously update user stats
    await db.users.updateOne(
      { _id: order.userId },
      { $inc: { totalOrders: 1 } }
    );
  }
});
```

**Summary:**

| Use Case | Solution |
|----------|----------|
| Single document update | Atomic operations (no transaction) |
| Financial transfers | Multi-document transaction |
| High-throughput writes | Avoid transactions |
| Eventual consistency OK | Change streams / 2PC |
| Complex multi-step | Transaction (keep short) |

---

## Security & Best Practices

### Q11: How do you secure a MongoDB deployment in production?

**Answer:**

**Multi-Layered Security Approach:**

**1. Authentication**
```javascript
// Enable authentication
// In mongod.conf:
security:
  authorization: enabled

// Create admin user
use admin
db.createUser({
  user: "admin",
  pwd: passwordPrompt(),  // Prompt for password
  roles: [ "root" ]
})

// Create application user with minimal privileges
use myapp
db.createUser({
  user: "app_user",
  pwd: passwordPrompt(),
  roles: [
    { role: "readWrite", db: "myapp" }
  ]
})

// Connect with authentication
mongosh --username app_user --password --authenticationDatabase myapp
```

**2. Network Security**
```javascript
// mongod.conf
net:
  bindIp: 127.0.0.1,10.0.0.5  // Don't bind to 0.0.0.0 (all interfaces)
  port: 27017
  tls:
    mode: requireTLS
    certificateKeyFile: /path/to/cert.pem
    CAFile: /path/to/ca.pem

// Firewall rules
// Only allow application servers to connect
iptables -A INPUT -p tcp --dport 27017 -s 10.0.0.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 27017 -j DROP
```

**3. Encryption**

**At Rest:**
```javascript
// mongod.conf
security:
  enableEncryption: true
  encryptionKeyFile: /path/to/keyfile

// All data files encrypted on disk
```

**In Transit (TLS):**
```javascript
// Connection string with TLS
mongodb://user:pass@host:27017/db?tls=true&tlsCAFile=/path/to/ca.pem
```

**Field-Level Encryption:**
```javascript
// Encrypt sensitive fields
const clientEncryption = new ClientEncryption(client, {
  keyVaultNamespace: 'encryption.__keyVault',
  kmsProviders: { local: { key: masterKey } }
});

const encrypted = await clientEncryption.encrypt(
  "123-45-6789",  // SSN
  { algorithm: "AEAD_AES_256_CBC_HMAC_SHA_512-Deterministic" }
);

await db.users.insertOne({
  name: "John Doe",
  ssn: encrypted  // Encrypted in database
});
```

**4. Role-Based Access Control (RBAC)**
```javascript
// Read-only user
db.createUser({
  user: "analyst",
  pwd: passwordPrompt(),
  roles: [ { role: "read", db: "analytics" } ]
})

// Custom role for specific permissions
db.createRole({
  role: "orderManager",
  privileges: [
    {
      resource: { db: "myapp", collection: "orders" },
      actions: [ "find", "update", "insert" ]
    },
    {
      resource: { db: "myapp", collection: "customers" },
      actions: [ "find" ]  // Read-only for customers
    }
  ],
  roles: []
})

// Assign custom role
db.createUser({
  user: "orderapp",
  pwd: passwordPrompt(),
  roles: [ "orderManager" ]
})
```

**5. Auditing**
```javascript
// mongod.conf
auditLog:
  destination: file
  format: JSON
  path: /var/log/mongodb/audit.json
  filter: |
    {
      atype: { 
        $in: [ 
          "authenticate", 
          "createUser", 
          "dropUser",
          "dropDatabase",
          "dropCollection"
        ]
      }
    }

// Audit log entry:
{
  "atype": "authenticate",
  "ts": ISODate("2024-01-20T10:30:00.000Z"),
  "local": { "ip": "10.0.0.5", "port": 27017 },
  "remote": { "ip": "10.0.0.10", "port": 52340 },
  "users": [ { "user": "app_user", "db": "myapp" } ],
  "result": 0  // 0 = success
}
```

**6. IP Whitelisting**
```javascript
// MongoDB Atlas: Configure IP Access List
// Allow only specific IPs/ranges
10.0.0.0/24  // Application servers
203.0.113.5  // Office IP

// Deny all others
```

**7. Least Privilege Principle**
```javascript
// Don't use root/admin for applications
// ❌ Bad
db.createUser({
  user: "myapp",
  pwd: "password",
  roles: [ "root" ]  // Too much access!
})

// ✅ Good
db.createUser({
  user: "myapp",
  pwd: "password",
  roles: [
    { role: "readWrite", db: "myapp" },  // Only what's needed
    { role: "read", db: "analytics" }
  ]
})
```

**8. Secrets Management**
```javascript
// Don't hardcode credentials
// ❌ Bad
const uri = "mongodb://user:password123@host:27017/db";

// ✅ Good - Use environment variables
const uri = `mongodb://${process.env.DB_USER}:${process.env.DB_PASS}@${process.env.DB_HOST}/${process.env.DB_NAME}`;

// ✅ Better - Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
const secrets = await getSecretsFromVault();
const uri = `mongodb://${secrets.username}:${secrets.password}@${secrets.host}/${secrets.database}`;
```

**9. Input Validation**
```javascript
// Prevent NoSQL injection
// ❌ Bad
const userId = req.query.userId;  // Could be malicious object
db.users.findOne({ _id: userId });

// ✅ Good
const userId = String(req.query.userId);  // Sanitize
if (!/^[a-f0-9]{24}$/.test(userId)) {
  throw new Error("Invalid user ID");
}
db.users.findOne({ _id: new ObjectId(userId) });

// Use schema validation
db.createCollection("users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "email"],
      properties: {
        name: { bsonType: "string", minLength: 1, maxLength: 100 },
        email: { bsonType: "string", pattern: "^.+@.+$" },
        age: { bsonType: "int", minimum: 0, maximum: 150 }
      }
    }
  }
})
```

**10. Regular Backups**
```javascript
// Automated backups with mongodump
mongodump --uri="mongodb://user:pass@host/db" --out=/backups/$(date +%Y%m%d)

// Verify backups regularly
mongorestore --uri="mongodb://localhost/test_restore" --dir=/backups/20240120

// Use MongoDB Atlas automated backups
// - Point-in-time recovery
// - Encrypted backups
// - Cross-region replication
```

**11. Monitoring & Alerts**
```javascript
// Monitor for security events
db.adminCommand({ getLog: "global" })

// Alert on:
// - Failed authentication attempts
// - Unusual query patterns
// - Large data exports
// - Schema changes
// - Privilege escalation attempts

// Example: Monitor failed logins
const failedLogins = db.system.profile.find({
  op: "command",
  command: { authenticate: 1 },
  errCode: { $exists: true }
}).count();

if (failedLogins > 10) {
  alertSecurityTeam("Multiple failed login attempts");
}
```

**12. Keep MongoDB Updated**
```bash
# Regular security patches
# Subscribe to MongoDB security announcements
# https://www.mongodb.com/alerts

# Upgrade process:
# 1. Test in staging
# 2. Backup production
# 3. Rolling upgrade (secondaries first, then primary)
```

**Security Checklist:**
- ✅ Authentication enabled
- ✅ TLS/SSL for all connections
- ✅ Encryption at rest
- ✅ IP whitelisting
- ✅ RBAC with least privilege
- ✅ Audit logging enabled
- ✅ Regular backups tested
- ✅ Monitoring and alerts
- ✅ Input validation
- ✅ Secrets management
- ✅ Regular security updates
- ✅ Network isolation (VPC/firewall)

---

## Final Thoughts

**What Makes a Senior MongoDB Developer:**

✅ Understands when to embed vs reference
✅ Designs schemas based on access patterns
✅ Creates efficient indexes following ESR rule
✅ Optimizes aggregation pipelines
✅ Knows when to use transactions (and when not to)
✅ Can design sharding strategy
✅ Implements proper security measures
✅ Monitors and optimizes performance
✅ Has experience with production migrations
✅ Thinks about operational concerns

**Red Flags:**
❌ Treats MongoDB like SQL (over-normalization, excessive $lookup)
❌ Doesn't consider query patterns in schema design
❌ Creates indexes without understanding impact
❌ Uses transactions everywhere
❌ No experience with production systems
❌ Doesn't know how to debug slow queries

---

## Additional Quick Questions

### Q12: What's the maximum document size in MongoDB?
**A:** 16MB. Use GridFS for larger files.

### Q13: What's the difference between updateOne() and replaceOne()?
**A:** `updateOne()` modifies specific fields using operators like $set. `replaceOne()` replaces the entire document (except _id).

### Q14: How do you handle duplicate key errors?
**A:** Use `upsert` option or catch the error code (11000) and handle appropriately.

### Q15: What are TTL indexes?
**A:** Time-To-Live indexes automatically delete documents after a specified time, useful for sessions, logs, temporary data.

```javascript
db.sessions.createIndex(
  { createdAt: 1 },
  { expireAfterSeconds: 3600 }  // Delete after 1 hour
)
```