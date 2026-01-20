# MongoDB Senior Developer Interview - Case-Based Discussion

## Case Study: E-Commerce Platform with High Traffic

**Scenario**: You're building a large-scale e-commerce platform. The system needs to handle millions of products, user sessions, orders, and real-time inventory updates.

---

## Part 1: Schema Design Discussion

### Opening Question
*"Let's start with the products catalog. How would you design the schema for products that have variants (like size, color)? Walk me through your thought process."*

**What to Look For:**
- **Embedded vs Referenced**: Do they understand when to embed vs reference?
- **Flexibility**: Can they handle varying product attributes?
- **Query patterns**: Do they consider how data will be queried?

**Good Answer Pattern:**
```javascript
// Embedded approach for variants (better for most e-commerce)
{
  _id: ObjectId("..."),
  name: "Cotton T-Shirt",
  category: "Apparel",
  basePrice: 29.99,
  variants: [
    { sku: "TSHIRT-RED-M", color: "Red", size: "M", stock: 50, priceModifier: 0 },
    { sku: "TSHIRT-BLUE-L", color: "Blue", size: "L", stock: 30, priceModifier: 2 }
  ],
  attributes: {
    material: "100% Cotton",
    brand: "BrandName"
  },
  tags: ["casual", "summer"],
  createdAt: ISODate("..."),
  updatedAt: ISODate("...")
}
```

**Key Points They Should Mention:**
- Embed variants because they're always queried together
- Keep SKU at variant level for inventory tracking
- Use arrays for tags to support multi-tag searches
- Flexible attributes object for product-specific fields

---

### Follow-up 1: "What if we need to show category hierarchies?"

**What to Look For:**
- Understanding of hierarchical data patterns
- Performance considerations

**Good Answer:**
"I'd use the **Materialized Path** or **Array of Ancestors** pattern:

```javascript
// Array of Ancestors pattern
{
  _id: ObjectId("..."),
  name: "Running Shoes",
  category: "Men's Athletic Footwear",
  ancestors: [
    { _id: "cat1", name: "Footwear" },
    { _id: "cat2", name: "Athletic Footwear" },
    { _id: "cat3", name: "Men's Athletic Footwear" }
  ],
  path: "Footwear,Athletic Footwear,Men's Athletic Footwear"
}
```

This lets us:
- Query all products in a category hierarchy efficiently
- Display breadcrumbs without extra queries
- Index on `ancestors._id` for fast lookups"

---

### Follow-up 2: "How would you handle inventory updates with high concurrency?"

**What to Look For:**
- Understanding of atomic operations
- Concurrency handling
- Race condition awareness

**Good Answer:**
"I'd use atomic operations with **findAndModify** or **updateOne** with conditions:

```javascript
// Atomic decrement with validation
db.products.updateOne(
  { 
    "variants.sku": "TSHIRT-RED-M",
    "variants.stock": { $gte: 1 }  // Only if stock available
  },
  { 
    $inc: { "variants.$.stock": -1 }  // Atomic decrement
  }
)
```

For high-traffic scenarios:
- Use **optimistic locking** with version numbers
- Consider **pre-allocating inventory** to carts with TTL
- Implement **distributed locks** for critical operations using MongoDB transactions"

---

## Part 2: Performance & Indexing

### Question
*"Your product search is getting slow. Users search by name, category, price range, and tags. How do you approach this?"*

**What to Look For:**
- Index strategy knowledge
- Understanding of compound indexes
- ESR rule awareness

**Good Answer:**
"I'd create a **compound index** following the ESR rule:
- **E**quality fields first
- **S**ort fields next
- **R**ange fields last

```javascript
// For query: category + tags + price range + sort by price
db.products.createIndex({
  category: 1,        // Equality
  tags: 1,            // Equality (can match multiple)
  basePrice: 1        // Range
})

// For text search
db.products.createIndex({
  name: "text",
  "attributes.brand": "text"
})
```

I'd also:
- Monitor with **explain()** to check index usage
- Use **hint()** if query planner picks wrong index
- Consider **partial indexes** for active products only
- Set up **covered queries** for common searches"

---

### Follow-up 1: "The explain() shows IXSCAN but it's still slow. What next?"

**Good Answer:**
"Several things to check:

1. **Index selectivity**: The index might not be selective enough
```javascript
db.products.aggregate([
  { $group: { _id: "$category", count: { $sum: 1 } } }
])
// If distribution is poor, index won't help much
```

2. **Covered queries**: Ensure all fields are in the index
```javascript
db.products.find(
  { category: "Electronics" },
  { _id: 0, name: 1, basePrice: 1 }  // Only indexed fields
)
```

3. **Working set size**: Check if indexes fit in RAM
```javascript
db.serverStatus().wiredTiger.cache
```

4. **Read preference**: For analytics, use secondaries to offload"

---

## Part 3: Aggregation Framework

### Question
*"Marketing wants a daily report showing top 10 products by revenue, grouped by category. How would you build this?"*

**What to Look For:**
- Aggregation pipeline knowledge
- Performance optimization
- Stage ordering

**Good Answer:**
```javascript
db.orders.aggregate([
  // Stage 1: Filter recent orders
  {
    $match: {
      status: "completed",
      createdAt: { $gte: new Date(Date.now() - 24*60*60*1000) }
    }
  },
  
  // Stage 2: Unwind order items
  { $unwind: "$items" },
  
  // Stage 3: Lookup product details
  {
    $lookup: {
      from: "products",
      localField: "items.productId",
      foreignField: "_id",
      as: "product"
    }
  },
  { $unwind: "$product" },
  
  // Stage 4: Group by category and product
  {
    $group: {
      _id: {
        category: "$product.category",
        productId: "$product._id",
        productName: "$product.name"
      },
      totalRevenue: { $sum: { $multiply: ["$items.quantity", "$items.price"] } },
      totalQuantity: { $sum: "$items.quantity" }
    }
  },
  
  // Stage 5: Sort within category
  { $sort: { "_id.category": 1, totalRevenue: -1 } },
  
  // Stage 6: Group by category and get top 10
  {
    $group: {
      _id: "$_id.category",
      topProducts: {
        $push: {
          productId: "$_id.productId",
          name: "$_id.productName",
          revenue: "$totalRevenue",
          quantity: "$totalQuantity"
        }
      }
    }
  },
  
  // Stage 7: Limit to top 10 per category
  {
    $project: {
      category: "$_id",
      topProducts: { $slice: ["$topProducts", 10] }
    }
  }
])
```

**Key optimizations they should mention:**
- Match early to reduce documents
- Use indexes on match fields
- Consider **materialized views** for frequently run reports
- Could use **$merge** to save results

---

## Part 4: Replication & Sharding

### Question
*"Your database is growing. Single server can't handle the load. Walk me through scaling this system."*

**What to Look For:**
- Understanding of horizontal vs vertical scaling
- Replication concepts
- Sharding strategy

**Good Answer:**
"I'd approach this in stages:

**Stage 1: Replication**
- Set up a **replica set** (1 primary, 2 secondaries)
- Distribute reads to secondaries for analytics
- Benefits: High availability, read scaling

**Stage 2: Assess Sharding Need**
Check if we really need sharding:
- Data size > RAM
- Write throughput > single server capacity
- Geographic distribution needed

**Stage 3: Choose Shard Key**
For e-commerce, I'd consider:

```javascript
// Option 1: Hashed _id (even distribution)
sh.shardCollection("store.products", { _id: "hashed" })
// Pros: Even distribution
// Cons: No range queries, no targeting

// Option 2: Category + _id compound (better for queries)
sh.shardCollection("store.products", { category: 1, _id: 1 })
// Pros: Category queries hit fewer shards
// Cons: Uneven if categories vary in size

// For orders: userId + createdAt
sh.shardCollection("store.orders", { userId: 1, createdAt: 1 })
// Pros: User queries targeted, time-based archival
```

**Stage 4: Zone Sharding**
For geographic distribution:
```javascript
sh.addShardToZone("shard0", "US")
sh.addShardToZone("shard1", "EU")
sh.updateZoneKeyRange(
  "store.orders",
  { region: "US", userId: MinKey },
  { region: "US", userId: MaxKey },
  "US"
)
```
"

---

### Follow-up: "What issues might you face with sharding?"

**Good Answer:**
"Several challenges:

1. **Jumbo chunks**: If shard key isn't granular
   - Solution: Choose high-cardinality key
   
2. **Hot shards**: Uneven distribution
   - Monitor with `sh.status()`
   - May need to refine shard key
   
3. **Scatter-gather queries**: Queries without shard key hit all shards
   - Include shard key in queries when possible
   
4. **Orphaned documents**: After failed migrations
   - Run `cleanupOrphaned` command
   
5. **Transactions across shards**: Expensive
   - Design schema to avoid when possible"

---

## Part 5: Transactions & Consistency

### Question
*"A user places an order. You need to: decrease inventory, create order, update user's order history. How do you ensure data consistency?"*

**What to Look For:**
- Transaction knowledge
- Error handling
- Performance awareness

**Good Answer:**
```javascript
const session = client.startSession();

try {
  await session.withTransaction(async () => {
    
    // 1. Reserve inventory
    const inventoryUpdate = await db.products.updateOne(
      { 
        "variants.sku": orderItem.sku,
        "variants.stock": { $gte: orderItem.quantity }
      },
      { $inc: { "variants.$.stock": -orderItem.quantity } },
      { session }
    );
    
    if (inventoryUpdate.modifiedCount === 0) {
      throw new Error("Insufficient inventory");
    }
    
    // 2. Create order
    const order = await db.orders.insertOne({
      userId: user._id,
      items: [orderItem],
      total: calculateTotal(orderItem),
      status: "pending",
      createdAt: new Date()
    }, { session });
    
    // 3. Update user history
    await db.users.updateOne(
      { _id: user._id },
      { 
        $push: { orderHistory: order.insertedId },
        $inc: { totalOrders: 1 }
      },
      { session }
    );
    
  });
  
} catch (error) {
  console.error("Transaction failed:", error);
  // All operations rolled back automatically
} finally {
  await session.endSession();
}
```

**Alternative approach for better performance:**
- Use **two-phase commits** pattern for cross-collection updates
- Implement **compensating transactions** for eventual consistency
- Consider **change streams** for async processing"

---

## Part 6: Monitoring & Operations

### Question
*"The app is slow during peak hours. How do you diagnose and fix it?"*

**What to Look For:**
- Monitoring tools knowledge
- Debugging methodology
- Proactive optimization

**Good Answer:**
"**Step 1: Identify the bottleneck**

```javascript
// Check slow queries
db.setProfilingLevel(1, { slowms: 100 })
db.system.profile.find().sort({ ts: -1 }).limit(5)

// Check current operations
db.currentOp({
  "active": true,
  "secs_running": { $gte: 3 }
})

// Server status
db.serverStatus().opcounters
db.serverStatus().connections
```

**Step 2: Common issues and fixes**

1. **Missing indexes**
   - Check explain plans
   - Add appropriate indexes

2. **Lock contention**
   - Check `db.serverStatus().locks`
   - Reduce transaction scope
   - Batch operations

3. **Memory issues**
   - Check working set vs RAM
   - Add RAM or scale horizontally

4. **Network latency**
   - Use connection pooling
   - Deploy near application servers
   - Use projection to reduce data transfer

**Step 3: Implement monitoring**
- Set up **MongoDB Atlas** or **Ops Manager**
- Configure alerts for:
  - Slow queries (>100ms)
  - High CPU/memory usage
  - Replication lag
  - Connection pool exhaustion"

---

## Part 7: Security

### Question
*"How do you secure this MongoDB deployment?"*

**Good Answer:**
"Multi-layered approach:

**1. Authentication & Authorization**
```javascript
// Create admin user
use admin
db.createUser({
  user: "admin",
  pwd: "securePassword",
  roles: ["root"]
})

// Application user with minimal privileges
use ecommerce
db.createUser({
  user: "app_user",
  pwd: "appPassword",
  roles: [
    { role: "readWrite", db: "ecommerce" }
  ]
})
```

**2. Network Security**
- Enable **TLS/SSL** for all connections
- Whitelist IP addresses
- Use **VPC** in cloud deployments
- Disable direct internet access

**3. Encryption**
- **Encryption at rest** using WiredTiger
- **Encryption in transit** with TLS
- Field-level encryption for PII:

```javascript
// Encrypt sensitive fields
{
  userId: ObjectId("..."),
  name: "John Doe",
  email: encrypt("john@example.com"),  // Encrypted
  creditCard: encrypt("1234-5678..."),  // Encrypted
  address: { /* normal */ }
}
```

**4. Auditing**
```javascript
// Enable auditing
auditLog:
  destination: file
  format: JSON
  path: /var/log/mongodb/audit.json
  filter: '{ atype: { $in: ["authenticate", "createUser"] } }'
```

**5. Regular maintenance**
- Keep MongoDB updated
- Regular security patches
- Principle of least privilege
- Regular backup verification"

---

## Evaluation Rubric

### Excellent (Senior Level)
- ✅ Explains trade-offs between different approaches
- ✅ Considers performance implications upfront
- ✅ Mentions monitoring and operational concerns
- ✅ Understands distributed systems challenges
- ✅ Provides specific examples and code
- ✅ Thinks about edge cases and failure scenarios

### Good (Mid-Senior Level)
- ✅ Solid understanding of MongoDB features
- ✅ Can design appropriate schemas
- ✅ Knows indexing strategies
- ⚠️ May need prompting for advanced topics
- ⚠️ Less experience with production issues

### Needs Improvement
- ❌ Doesn't consider query patterns in schema design
- ❌ Unclear about when to use transactions
- ❌ Limited knowledge of replication/sharding
- ❌ Can't explain trade-offs

---

## Additional Scenario-Based Questions

### Scenario 1: Data Migration
*"You need to migrate 100M documents to a new schema. How?"*

**Key Points:**
- Batch processing
- Zero-downtime strategy
- Dual writes during transition
- Validation

### Scenario 2: Time-Series Data
*"You need to store product view analytics (millions per day). Design?"*

**Key Points:**
- Time-series collections (MongoDB 5.0+)
- Bucketing pattern
- TTL indexes for auto-deletion
- Aggregation for analytics

### Scenario 3: Full-Text Search
*"Users need advanced search with typo tolerance. MongoDB or external?"*

**Key Points:**
- MongoDB Atlas Search vs Elasticsearch
- Trade-offs discussion
- Hybrid approach
- Sync strategies

---

## Tips for Interviewer

1. **Let them drive**: See how they approach problems
2. **Dig deeper**: Ask "why" to understand reasoning
3. **Challenge assumptions**: "What if traffic is 10x higher?"
4. **Real scenarios**: Share actual problems your team faces
5. **Code together**: Open MongoDB shell and explore together
6. **Be collaborative**: This should feel like pair programming

Good luck with your interview! 🚀