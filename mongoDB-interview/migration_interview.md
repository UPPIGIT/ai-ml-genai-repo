# MongoDB Migration Interview - Technical Discussion Guide

## Case Study: Large-Scale System Migration from Relational to MongoDB

**Scenario**: We're planning to migrate a complex enterprise system with multiple interconnected modules to MongoDB. The system has been running on a traditional relational database for years and handles millions of transactions.

---

## Part 1: Migration Strategy & Planning

### Opening Discussion
*"Imagine you're leading a migration from a relational database to MongoDB for a large enterprise system. What would be your overall approach and what concerns would you have upfront?"*

**What to Look For:**
- Systematic thinking
- Risk awareness
- Experience with migrations
- Business continuity focus

**Strong Answer:**
"I'd approach this in phases rather than a big-bang migration:

**Phase 1: Assessment (2-3 weeks)**
- Analyze current schema and relationships
- Identify query patterns and bottlenecks
- Map dependencies between modules
- Assess data volume and growth rate
- Review application code for DB interactions

**Phase 2: Proof of Concept (3-4 weeks)**
- Choose 1-2 non-critical modules to migrate first
- Design MongoDB schema for those modules
- Test performance with production-like data volume
- Validate the migration approach

**Phase 3: Dual-Write Strategy**
- Implement writes to both databases
- Validate data consistency
- Allows rollback if issues occur

**Phase 4: Gradual Migration**
- Migrate module by module
- Start with read-only or less critical modules
- Monitor performance and fix issues
- Finally migrate core transactional modules

**Phase 5: Cutover & Decommission**
- Switch reads to MongoDB
- Monitor for 2-4 weeks
- Decommission old database

**Key Concerns:**
- Data consistency during migration
- Zero downtime requirement
- Training team on MongoDB
- Rollback strategy if issues arise
- Performance validation before go-live"

---

### Follow-up 1: "How do you decide what to migrate first?"

**Good Answer:**
"I'd use a prioritization matrix:

**Migrate First:**
✅ **Low-risk, high-value modules**
- Read-heavy reporting modules
- Log/audit tables (perfect for MongoDB)
- Configuration/metadata stores
- Session management

✅ **Modules that benefit most from MongoDB**
- JSON/flexible schema requirements
- Nested/hierarchical data
- High write throughput needs
- Need for horizontal scaling

**Migrate Later:**
✅ **High-risk modules**
- Core transaction processing
- Modules with complex joins
- Financial/accounting modules (after validation)
- Modules with strict ACID requirements

**Example Priority:**
```
High Priority:
1. Activity Logs (append-only, no joins)
2. User Preferences (document-oriented)
3. Product Catalog (hierarchical, flexible)

Medium Priority:
4. Order History (some joins, mostly reads)
5. Inventory Management (real-time updates)

Low Priority (last):
6. Payment Processing (ACID critical)
7. Accounting Ledger (strong consistency needs)
```"

---

### Follow-up 2: "How do you handle referential integrity during migration?"

**Strong Answer:**
"Relational databases enforce referential integrity at the database level. In MongoDB, we need to handle this differently:

**Strategy 1: Application-Level Validation**
```javascript
// Before inserting order, validate user exists
async function createOrder(orderData) {
  // Validate references exist
  const user = await db.users.findOne({ _id: orderData.userId });
  if (!user) {
    throw new Error('Invalid user reference');
  }
  
  const product = await db.products.findOne({ _id: orderData.productId });
  if (!product) {
    throw new Error('Invalid product reference');
  }
  
  // Now safe to create order
  return await db.orders.insertOne(orderData);
}
```

**Strategy 2: Denormalization (Recommended)**
```javascript
// Instead of references, embed critical data
{
  _id: ObjectId("..."),
  orderId: "ORD-12345",
  user: {
    userId: ObjectId("..."),
    name: "John Doe",
    email: "john@example.com"  // Snapshot at order time
  },
  items: [{
    productId: ObjectId("..."),
    name: "Laptop",  // Snapshot
    price: 999.99    // Price at purchase time
  }],
  status: "completed"
}
```

**Strategy 3: Use Transactions for Critical Operations**
```javascript
const session = client.startSession();
try {
  await session.withTransaction(async () => {
    // Delete user
    await db.users.deleteOne({ _id: userId }, { session });
    
    // Delete related orders
    await db.orders.deleteMany({ userId: userId }, { session });
    
    // Delete related sessions
    await db.sessions.deleteMany({ userId: userId }, { session });
  });
} finally {
  await session.endSession();
}
```

**Strategy 4: Change Streams for Consistency**
```javascript
// Listen for deletions and cleanup references
const changeStream = db.users.watch([
  { $match: { operationType: 'delete' } }
]);

changeStream.on('change', async (change) => {
  const deletedUserId = change.documentKey._id;
  
  // Cleanup orphaned records
  await db.orders.updateMany(
    { userId: deletedUserId },
    { $set: { userDeleted: true } }
  );
});
```

**Migration Approach:**
- Map all foreign key relationships from SQL
- Decide: embed vs reference for each relationship
- Implement validation in application layer
- Add unit tests for referential integrity
- Use schema validation rules in MongoDB:

```javascript
db.createCollection('orders', {
  validator: {
    $jsonSchema: {
      required: ['userId', 'items', 'total'],
      properties: {
        userId: { bsonType: 'objectId' },
        items: { 
          bsonType: 'array',
          minItems: 1
        }
      }
    }
  }
})
```"

---

## Part 2: Schema Transformation

### Discussion Point
*"Let's say you have a typical relational schema with Users, Orders, OrderItems, Products, and Payments tables with foreign keys. Walk me through how you'd redesign this for MongoDB."*

**What to Look For:**
- Understanding of embedding vs referencing
- Query pattern consideration
- Denormalization comfort

**Strong Answer:**
"I'd analyze the access patterns first:

**SQL Schema Analysis:**
```sql
-- Relational structure
Users (id, name, email, address)
Orders (id, user_id, order_date, status, total)
OrderItems (id, order_id, product_id, quantity, price)
Products (id, name, category, price, stock)
Payments (id, order_id, amount, method, status, transaction_id)
```

**Access Pattern Analysis:**
1. **How is data read?**
   - Show order with all items (JOIN heavy)
   - Show user with order history
   - Product catalog browsing
   - Payment verification

2. **How is data written?**
   - Create order (inserts across 3 tables)
   - Update inventory
   - Process payment

**MongoDB Schema Design:**

```javascript
// Users Collection - Keep separate (authentication, profile management)
{
  _id: ObjectId("user123"),
  name: "John Doe",
  email: "john@example.com",
  password: "hashed_password",
  address: {
    street: "123 Main St",
    city: "New York",
    zip: "10001"
  },
  createdAt: ISODate("2024-01-15"),
  preferences: {
    newsletter: true,
    language: "en"
  }
}

// Products Collection - Keep separate (catalog management)
{
  _id: ObjectId("prod456"),
  sku: "LAPTOP-001",
  name: "MacBook Pro",
  category: "Electronics",
  price: 1999.99,
  stock: 50,
  attributes: {
    brand: "Apple",
    specs: {
      ram: "16GB",
      storage: "512GB"
    }
  },
  images: ["url1", "url2"],
  updatedAt: ISODate("2024-01-20")
}

// Orders Collection - EMBED items and payment (read together)
{
  _id: ObjectId("order789"),
  orderNumber: "ORD-2024-001",
  
  // Denormalized user info (snapshot at order time)
  user: {
    userId: ObjectId("user123"),
    name: "John Doe",
    email: "john@example.com",
    shippingAddress: {
      street: "123 Main St",
      city: "New York",
      zip: "10001"
    }
  },
  
  // EMBEDDED order items (always queried together)
  items: [
    {
      productId: ObjectId("prod456"),
      sku: "LAPTOP-001",
      name: "MacBook Pro",  // Snapshot
      quantity: 1,
      unitPrice: 1999.99,
      subtotal: 1999.99
    },
    {
      productId: ObjectId("prod457"),
      sku: "MOUSE-002",
      name: "Magic Mouse",
      quantity: 2,
      unitPrice: 79.99,
      subtotal: 159.98
    }
  ],
  
  // EMBEDDED payment info (part of order)
  payment: {
    method: "credit_card",
    status: "completed",
    transactionId: "txn_abc123",
    amount: 2159.97,
    processedAt: ISODate("2024-01-20T10:30:00Z")
  },
  
  status: "shipped",
  total: 2159.97,
  tax: 172.80,
  shipping: 15.00,
  
  createdAt: ISODate("2024-01-20T10:25:00Z"),
  updatedAt: ISODate("2024-01-21T14:00:00Z"),
  
  tracking: {
    carrier: "UPS",
    trackingNumber: "1Z999AA10123456784",
    shippedAt: ISODate("2024-01-21T14:00:00Z")
  }
}
```

**Why This Design:**

✅ **Orders embed items** - Always displayed together, reduces JOINs
✅ **Orders embed payment** - Part of same business transaction
✅ **Denormalize user/product snapshots** - Historical accuracy (price/address at order time)
✅ **Products separate** - Updated frequently, reused across orders
✅ **Users separate** - Authentication, profile updates independent

**Benefits Over SQL:**
- Single query to get complete order (no JOINs)
- Atomic updates to entire order
- Historical accuracy (prices don't change if product price changes)
- Scales horizontally easily

**Trade-offs:**
- Data duplication (user info, product snapshots)
- Need to update denormalized data if corrections needed
- Slightly larger document size"

---

### Follow-up: "How do you handle the case where product information changes after an order is placed?"

**Good Answer:**
"This is actually a feature, not a bug! In e-commerce, you want historical accuracy:

**Scenario:** Product price increases from $100 to $120

**In SQL (problem):**
```sql
-- Order references product_id
SELECT o.*, p.price FROM orders o 
JOIN products p ON o.product_id = p.id
-- Returns $120 even though customer paid $100!
```

**In MongoDB (correct):**
```javascript
// Order has snapshot of price at purchase time
{
  items: [{
    productId: ObjectId("..."),
    name: "Widget",
    unitPrice: 100.00  // Price when ordered - never changes
  }]
}
```

**For cases where you DO need to update:**

```javascript
// Example: Product recall - need to update name/description
db.orders.updateMany(
  { "items.productId": ObjectId("recalled_product") },
  { 
    $set: { 
      "items.$.recallNotice": "This product has been recalled",
      "items.$.updatedName": "Widget (RECALLED - DO NOT USE)"
    } 
  }
)

// Example: Tax law change - recalculate old orders
db.orders.updateMany(
  { 
    createdAt: { $gte: ISODate("2024-01-01") },
    status: "pending" 
  },
  [
    {
      $set: {
        tax: { $multiply: ["$subtotal", 0.08] },  // New tax rate
        total: { 
          $add: [
            "$subtotal", 
            { $multiply: ["$subtotal", 0.08] },
            "$shipping"
          ]
        }
      }
    }
  ]
)
```

**Best Practice:**
Keep both current reference AND historical snapshot:
```javascript
{
  items: [{
    productId: ObjectId("prod123"),  // For lookups/analytics
    snapshot: {                       // Historical accuracy
      name: "Widget",
      price: 100.00,
      sku: "WID-001"
    }
  }]
}
```"

---

## Part 3: Data Migration Execution

### Discussion Point
*"You have 50 million records to migrate. How do you execute the actual data migration without disrupting the live system?"*

**What to Look For:**
- Experience with large data migrations
- Understanding of ETL processes
- Risk mitigation strategies

**Strong Answer:**
"For a migration of this scale, I'd use a multi-phase approach:

**Phase 1: Historical Data Migration (Bulk)**

```javascript
// Migration script using cursor batching
async function migrateHistoricalOrders() {
  const batchSize = 1000;
  let processed = 0;
  
  // Read from SQL in batches
  const sqlCursor = await sqlDb.query(
    'SELECT * FROM orders WHERE created_at < NOW() - INTERVAL 30 DAY ORDER BY id'
  );
  
  let batch = [];
  
  for await (const sqlRow of sqlCursor) {
    // Transform SQL row to MongoDB document
    const mongoDoc = transformOrder(sqlRow);
    batch.push(mongoDoc);
    
    // Bulk insert when batch is full
    if (batch.length >= batchSize) {
      await db.orders.insertMany(batch, { ordered: false });
      processed += batch.length;
      batch = [];
      
      console.log(`Migrated ${processed} orders`);
      
      // Throttle to avoid overwhelming the system
      await sleep(100);
    }
  }
  
  // Insert remaining
  if (batch.length > 0) {
    await db.orders.insertMany(batch, { ordered: false });
  }
}

function transformOrder(sqlRow) {
  return {
    _id: new ObjectId(),
    orderNumber: sqlRow.order_number,
    user: {
      userId: new ObjectId(sqlRow.user_id),
      name: sqlRow.user_name,
      email: sqlRow.user_email
    },
    items: JSON.parse(sqlRow.items_json), // Fetched with JOIN
    total: parseFloat(sqlRow.total),
    createdAt: new Date(sqlRow.created_at),
    // ... map all fields
  };
}
```

**Phase 2: Dual-Write for New Data**

```javascript
// Application layer writes to both databases
async function createOrder(orderData) {
  const session = await mongoClient.startSession();
  const sqlTransaction = await sqlDb.beginTransaction();
  
  try {
    // Write to MongoDB
    const mongoResult = await db.orders.insertOne(orderData, { session });
    
    // Write to SQL
    const sqlResult = await sqlDb.query(
      'INSERT INTO orders (user_id, total, ...) VALUES (?, ?, ...)',
      [orderData.userId, orderData.total, ...]
    );
    
    // Commit both
    await session.commitTransaction();
    await sqlTransaction.commit();
    
    return mongoResult;
    
  } catch (error) {
    // Rollback both
    await session.abortTransaction();
    await sqlTransaction.rollback();
    throw error;
  } finally {
    await session.endSession();
  }
}
```

**Phase 3: Data Validation**

```javascript
// Validate migration accuracy
async function validateMigration() {
  // Count comparison
  const sqlCount = await sqlDb.query('SELECT COUNT(*) FROM orders');
  const mongoCount = await db.orders.countDocuments();
  
  console.log(`SQL: ${sqlCount}, MongoDB: ${mongoCount}`);
  
  // Sample validation
  const sampleSize = 10000;
  const samples = await sqlDb.query(
    'SELECT id FROM orders ORDER BY RAND() LIMIT ?',
    [sampleSize]
  );
  
  let mismatches = 0;
  
  for (const sample of samples) {
    const sqlOrder = await fetchSqlOrder(sample.id);
    const mongoOrder = await db.orders.findOne({ 
      orderNumber: sqlOrder.order_number 
    });
    
    if (!deepEqual(sqlOrder, mongoOrder)) {
      mismatches++;
      console.error('Mismatch found:', sample.id);
    }
  }
  
  console.log(`Validation: ${mismatches}/${sampleSize} mismatches`);
}
```

**Phase 4: Cutover Strategy**

```javascript
// Feature flag to control read source
class OrderService {
  async getOrder(orderId) {
    if (featureFlags.readFromMongo) {
      return await this.getOrderFromMongo(orderId);
    } else {
      return await this.getOrderFromSql(orderId);
    }
  }
  
  // Gradual rollout
  async getOrderWithCanary(orderId) {
    // 10% of traffic reads from MongoDB
    if (Math.random() < 0.10) {
      const mongoResult = await this.getOrderFromMongo(orderId);
      
      // Shadow read from SQL to compare
      const sqlResult = await this.getOrderFromSql(orderId);
      
      // Log discrepancies
      if (!deepEqual(mongoResult, sqlResult)) {
        logger.error('Data mismatch', { orderId, mongoResult, sqlResult });
      }
      
      return mongoResult;
    } else {
      return await this.getOrderFromSql(orderId);
    }
  }
}
```

**Timeline for 50M Records:**

```
Week 1-2: Historical data migration (older records)
  - 1M records/day = 50 days if single-threaded
  - Parallel migration: 10M records/day = 5 days
  - Run during off-peak hours

Week 3-4: Dual-write implementation & testing
  - All new writes go to both systems
  - Validate consistency

Week 5-6: Catch-up migration
  - Migrate records created during Phase 1
  - Continuous validation

Week 7-8: Canary reads
  - 1% → 10% → 50% → 100% traffic to MongoDB
  - Monitor for issues

Week 9-10: Decommission SQL
  - Stop dual writes
  - Archive SQL database
```

**Key Considerations:**

✅ **Parallel Processing**
```javascript
// Use multiple workers
const workers = 10;
const chunks = splitIntoChunks(totalRecords, workers);

await Promise.all(
  chunks.map(chunk => migrateChunk(chunk))
);
```

✅ **Idempotency**
```javascript
// Safe to re-run if migration fails
await db.orders.updateOne(
  { orderNumber: sqlRow.order_number },
  { $setOnInsert: transformedDoc },
  { upsert: true }
);
```

✅ **Progress Tracking**
```javascript
// Track migration progress
db.migration_status.updateOne(
  { type: 'orders' },
  { 
    $set: { 
      lastMigratedId: currentId,
      count: processedCount,
      updatedAt: new Date()
    }
  },
  { upsert: true }
);
```

✅ **Monitoring & Rollback**
```javascript
// Quick rollback if issues found
if (errorRate > threshold) {
  featureFlags.readFromMongo = false;
  featureFlags.writeToMongo = false;
  alertTeam('Rolling back migration');
}
```"

---

## Part 4: Query Optimization & Performance

### Discussion Point
*"After migration, some queries are slower than they were in SQL. How do you approach optimization?"*

**Strong Answer:**
"First, I'd identify which queries are slow and why:

**Step 1: Enable Profiling**
```javascript
// Profile slow queries (>100ms)
db.setProfilingLevel(1, { slowms: 100 });

// Review slow queries
db.system.profile.find({
  millis: { $gt: 100 }
}).sort({ ts: -1 }).limit(10).pretty();

// Look for patterns
db.system.profile.aggregate([
  { $match: { millis: { $gt: 100 } } },
  { $group: {
    _id: "$command.find",
    count: { $sum: 1 },
    avgMs: { $avg: "$millis" },
    maxMs: { $max: "$millis" }
  }},
  { $sort: { count: -1 } }
]);
```

**Step 2: Analyze Specific Queries**

**Example 1: SQL JOIN converted to lookup**

❌ **Slow Approach:**
```javascript
// This mimics SQL JOIN but is slow
const orders = await db.orders.find({ userId: userId }).toArray();

for (const order of orders) {
  const user = await db.users.findOne({ _id: order.userId });
  const items = await db.orderItems.find({ orderId: order._id }).toArray();
  order.user = user;
  order.items = items;
}
```

✅ **Fast Approach (Aggregation):**
```javascript
db.orders.aggregate([
  { $match: { userId: ObjectId(userId) } },
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  },
  { $unwind: "$user" },
  {
    $lookup: {
      from: "orderItems",
      localField: "_id",
      foreignField: "orderId",
      as: "items"
    }
  }
]);
```

✅ **Better: Redesign Schema (Embed)**
```javascript
// No lookup needed - items already embedded
db.orders.find({ userId: ObjectId(userId) });
```

**Example 2: Missing Index**

❌ **Slow Query:**
```javascript
db.orders.find({
  "user.email": "john@example.com",
  status: "pending",
  createdAt: { $gte: ISODate("2024-01-01") }
}).explain("executionStats");

// Result: COLLSCAN (scans all documents)
```

✅ **Add Compound Index:**
```javascript
// Create index following ESR rule
db.orders.createIndex({
  "user.email": 1,    // Equality
  status: 1,           // Equality
  createdAt: -1        // Range
});

// Verify index usage
db.orders.find({
  "user.email": "john@example.com",
  status: "pending",
  createdAt: { $gte: ISODate("2024-01-01") }
}).explain("executionStats");

// Result: IXSCAN with low documents examined
```

**Example 3: Inefficient Aggregation**

❌ **Slow:**
```javascript
// Lookup before match (processes all docs)
db.orders.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  },
  { $match: { "user.country": "US" } }
]);
```

✅ **Optimized:**
```javascript
// Match first (reduces docs for lookup)
db.orders.aggregate([
  { $match: { status: "completed" } },  // Filter early
  {
    $lookup: {
      from: "users",
      let: { userId: "$userId" },
      pipeline: [
        { $match: { 
          $expr: { $eq: ["$_id", "$$userId"] },
          country: "US"  // Filter in subpipeline
        }}
      ],
      as: "user"
    }
  },
  { $match: { user: { $ne: [] } } }  // Only matched users
]);
```

**Step 3: Common Migration Performance Issues**

**Issue 1: Over-normalization**
```javascript
// Migrated as-is from SQL (requires multiple queries)
❌ Users collection
❌ Orders collection (userId reference)
❌ OrderItems collection (orderId reference)
❌ Products collection (productId reference)

// Redesign: Embed what's queried together
✅ Orders collection with embedded items and user snapshot
```

**Issue 2: Index Strategy**

**In SQL:**
```sql
-- Indexes created automatically for foreign keys
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
```

**In MongoDB: Must explicitly create**
```javascript
// Create indexes for common queries
db.orders.createIndex({ userId: 1, createdAt: -1 });
db.orders.createIndex({ "user.email": 1 });
db.orders.createIndex({ status: 1, createdAt: -1 });
db.orders.createIndex({ orderNumber: 1 }, { unique: true });

// For embedded arrays
db.orders.createIndex({ "items.productId": 1 });
db.orders.createIndex({ "items.sku": 1 });
```

**Issue 3: Large Document Scans**

```javascript
// Don't fetch entire documents if not needed
❌ db.orders.find({ status: "pending" });

✅ db.orders.find(
  { status: "pending" },
  { orderNumber: 1, total: 1, createdAt: 1 }  // Projection
);

// Covered query (index-only)
db.orders.find(
  { status: "pending" },
  { _id: 0, orderNumber: 1, total: 1 }
).hint({ status: 1, orderNumber: 1, total: 1 });
```

**Step 4: Performance Testing**

```javascript
// Benchmark before/after optimization
async function benchmarkQuery() {
  const iterations = 100;
  const start = Date.now();
  
  for (let i = 0; i < iterations; i++) {
    await db.orders.find({
      status: "pending",
      createdAt: { $gte: new Date("2024-01-01") }
    }).toArray();
  }
  
  const duration = Date.now() - start;
  console.log(`Average: ${duration/iterations}ms per query`);
}
```

**Step 5: Comparison with SQL Performance**

```javascript
// Track query performance migration
const metrics = {
  sql: {
    getUserOrders: 45,  // ms
    searchProducts: 120,
    createOrder: 85
  },
  mongodb: {
    getUserOrders: 12,  // Much faster (no joins)
    searchProducts: 150, // Slower (needs index)
    createOrder: 20     // Much faster (single doc)
  }
};

// Goal: Match or beat SQL performance for all queries
```"

---

## Part 5: Handling Complex Relationships

### Discussion Point
*"The system has many-to-many relationships. For example, users can belong to multiple organizations, and organizations have multiple users with different roles. How do you model this?"*

**Strong Answer:**
"Many-to-many relationships are one of the trickiest parts of migrating from SQL. Let me show different approaches:

**In SQL:**
```sql
Users (id, name, email)
Organizations (id, name, type)
UserOrganizations (user_id, organization_id, role, joined_date)
```

**MongoDB Approach 1: Embed Arrays in Both Directions**

```javascript
// Users collection
{
  _id: ObjectId("user123"),
  name: "John Doe",
  email: "john@example.com",
  organizations: [
    {
      orgId: ObjectId("org456"),
      orgName: "Acme Corp",  // Denormalized
      role: "admin",
      joinedDate: ISODate("2023-01-15")
    },
    {
      orgId: ObjectId("org789"),
      orgName: "Tech Inc",
      role: "member",
      joinedDate: ISODate("2023-06-20")
    }
  ]
}

// Organizations collection
{
  _id: ObjectId("org456"),
  name: "Acme Corp",
  type: "enterprise",
  memberIds: [ObjectId("user123"), ObjectId("user456"), ...]
}

// UserOrganizations collection - full relationship data
{
  _id: ObjectId("..."),
  userId: ObjectId("user123"),
  organizationId: ObjectId("org456"),
  role: "admin",
  permissions: ["manage_users", "view_reports", "edit_settings"],
  joinedDate: ISODate("2023-01-15"),
  invitedBy: ObjectId("user000"),
  lastActive: ISODate("2024-01-20")
}
```

**Query Patterns:**

```javascript
// Quick check: Is user in org?
const user = await db.users.findOne({ 
  _id: userId,
  "organizations.orgId": orgId 
});

// Get user's role in org
const user = await db.users.findOne(
  { _id: userId },
  { organizations: { $elemMatch: { orgId: orgId } } }
);

// Get full relationship details
const membership = await db.userOrganizations.findOne({
  userId: userId,
  organizationId: orgId
});

// Get all org members with details
db.userOrganizations.aggregate([
  { $match: { organizationId: orgId } },
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "userDetails"
    }
  },
  { $unwind: "$userDetails" },
  {
    $project: {
      userId: 1,
      role: 1,
      permissions: 1,
      name: "$userDetails.name",
      email: "$userDetails.email"
    }
  }
]);
```

**Update Operations:**

```javascript
// Add user to organization
async function addUserToOrg(userId, orgId, role) {
  const session = client.startSession();
  
  try {
    await session.withTransaction(async () => {
      // 1. Add to user's organizations array
      await db.users.updateOne(
        { _id: userId },
        { 
          $addToSet: { 
            organizations: { orgId: orgId, role: role } 
          } 
        },
        { session }
      );
      
      // 2. Add to org's member list
      await db.organizations.updateOne(
        { _id: orgId },
        { 
          $addToSet: { memberIds: userId },
          $inc: { memberCount: 1 }
        },
        { session }
      );
      
      // 3. Create full relationship record
      await db.userOrganizations.insertOne({
        userId: userId,
        organizationId: orgId,
        role: role,
        permissions: getDefaultPermissions(role),
        joinedDate: new Date()
      }, { session });
    });
  } finally {
    await session.endSession();
  }
}

// Update role
async function updateUserRole(userId, orgId, newRole) {
  const session = client.startSession();
  
  try {
    await session.withTransaction(async () => {
      // Update in users collection
      await db.users.updateOne(
        { 
          _id: userId,
          "organizations.orgId": orgId 
        },
        { 
          $set: { "organizations.$.role": newRole } 
        },
        { session }
      );
      
      // Update in relationship collection
      await db.userOrganizations.updateOne(
        { userId: userId, organizationId: orgId },
        { 
          $set: { 
            role: newRole,
            permissions: getDefaultPermissions(newRole)
          } 
        },
        { session }
      );
    });
  } finally {
    await session.endSession();
  }
}
```

**Why Hybrid Approach Works:**

✅ Fast common queries (user's orgs, basic checks)
✅ Detailed data when needed (permissions, audit trail)
✅ Manageable consistency with transactions
✅ Scales well for most use cases"

---

## Part 6: Transaction Management

### Discussion Point
*"The system has critical workflows that span multiple operations. How do you ensure data consistency in MongoDB compared to SQL transactions?"*

**Strong Answer:**
"MongoDB has ACID transactions since version 4.0, but they should be used thoughtfully:

**Understanding MongoDB Transactions:**

**Single Document Operations (Atomic by Default)**
```javascript
// This is automatically atomic - no transaction needed
db.accounts.updateOne(
  { _id: accountId },
  {
    $inc: { balance: -100 },
    $push: { 
      transactions: {
        type: "withdrawal",
        amount: 100,
        timestamp: new Date()
      }
    }
  }
);
```

**Multi-Document Transactions (When Needed)**
```javascript
// Transfer between accounts - needs transaction
async function transferMoney(fromAccount, toAccount, amount) {
  const session = client.startSession();
  
  try {
    await session.withTransaction(async () => {
      // Deduct from source
      const debitResult = await db.accounts.updateOne(
        { 
          _id: fromAccount,
          balance: { $gte: amount }  // Ensure sufficient funds
        },
        { $inc: { balance: -amount } },
        { session }
      );
      
      if (debitResult.modifiedCount === 0) {
        throw new Error("Insufficient funds");
      }
      
      // Add to destination
      await db.accounts.updateOne(
        { _id: toAccount },
        { $inc: { balance: amount } },
        { session }
      );
      
      // Record transaction
      await db.transactionLog.insertOne({
        from: fromAccount,
        to: toAccount,
        amount: amount,
        timestamp: new Date(),
        status: "completed"
      }, { session });
    });
    
    return { success: true };
    
  } catch (error) {
    console.error("Transaction failed:", error);
    return { success: false, error: error.message };
  } finally {
    await session.endSession();
  }
}
```

**Migration Strategy for Transactional Code:**

**SQL Transaction:**
```sql
BEGIN TRANSACTION;

INSERT INTO orders (user_id, total, status) VALUES (?, ?, ?);
SET @order_id = LAST_INSERT_ID();

INSERT INTO order_items (order_id, product_id, quantity, price) 
VALUES (@order_id, ?, ?, ?);

UPDATE products SET stock = stock - ? WHERE id = ?;

UPDATE users SET total_orders = total_orders + 1 WHERE id = ?;

COMMIT;
```

**MongoDB Equivalent (Redesigned):**

**Option 1: Embed Everything (Best)**
```javascript
// Single document operation - atomic by default
const order = {
  userId: ObjectId(userId),
  items: [{
    productId: ObjectId(productId),
    name: "Product Name",
    quantity: 2,
    price: 99.99
  }],
  total: 199.98,
  status: "pending",
  createdAt: new Date()
};

await db.orders.insertOne(order);

// Separate operation for inventory (if needed)
await db.products.updateOne(
  { _id: productId, stock: { $gte: 2 } },
  { $inc: { stock: -2 } }
);
```

**Option 2: Use Transaction (When Necessary)**
```javascript
async function createOrder(orderData) {
  const session = client.startSession();
  
  try {
    const result = await session.withTransaction(async () => {
      // 1. Check and reserve inventory
      const inventoryUpdate = await db.products.updateOne(
        { 
          _id: orderData.productId,
          stock: { $gte: orderData.quantity }
        },
        { $inc: { stock: -orderData.quantity } },
        { session }
      );
      
      if (inventoryUpdate.modifiedCount === 0) {
        throw new Error("Insufficient inventory");
      }
      
      // 2. Create order
      const orderResult = await db.orders.insertOne({
        userId: orderData.userId,
        items: orderData.items,
        total: orderData.total,
        status: "confirmed",
        createdAt: new Date()
      }, { session });
      
      // 3. Update user stats
      await db.users.updateOne(
        { _id: orderData.userId },
        { 
          $inc: { totalOrders: 1, totalSpent: orderData.total },
          $push: { 
            recentOrders: {
              $each: [orderResult.insertedId],
              $slice: -10  // Keep last 10
            }
          }
        },
        { session }
      );
      
      return orderResult;
    }, {
      readPreference: 'primary',
      readConcern: { level: 'snapshot' },
      writeConcern: { w: 'majority' }
    });
    
    return result;
    
  } catch (error) {
    console.error("Order creation failed:", error);
    throw error;
  } finally {
    await session.endSession();
  }
}
```

**Best Practices:**

✅ **Prefer Single Document Operations**
- Embed related data when possible
- Use atomic operators ($inc, $push, etc.)

✅ **Keep Transactions Short**
```javascript
// ❌ Bad - long-running transaction
await session.withTransaction(async () => {
  const data = await fetchExternalAPI();  // Slow!
  const processed = heavyProcessing(data);  // Slow!
  await db.collection.insertOne(processed, { session });
});

// ✅ Good - minimal transaction scope
const data = await fetchExternalAPI();
const processed = heavyProcessing(data);

await session.withTransaction(async () => {
  await db.collection.insertOne(processed, { session });
});
```

✅ **Handle Transaction Failures**
```javascript
const maxRetries = 3;
let attempt = 0;

while (attempt < maxRetries) {
  try {
    await session.withTransaction(async () => {
      // Transaction logic
    });
    break;  // Success
  } catch (error) {
    if (error.hasErrorLabel('TransientTransactionError')) {
      attempt++;
      await sleep(100 * attempt);  // Exponential backoff
    } else {
      throw error;  // Non-transient error
    }
  }
}
```

✅ **Consider Two-Phase Commit Pattern**
```javascript
// For distributed transactions across services
async function distributedOrder() {
  const transactionId = new ObjectId();
  
  // Phase 1: Prepare
  await db.orders.insertOne({
    _id: transactionId,
    status: "pending",
    ...orderData
  });
  
  await db.inventory.updateOne(
    { productId: productId },
    { 
      $inc: { reserved: quantity },
      $push: { pendingTransactions: transactionId }
    }
  );
  
  // Phase 2: Commit
  try {
    await processPayment();
    
    // Commit order
    await db.orders.updateOne(
      { _id: transactionId },
      { $set: { status: "confirmed" } }
    );
    
    // Commit inventory
    await db.inventory.updateOne(
      { productId: productId },
      { 
        $inc: { stock: -quantity, reserved: -quantity },
        $pull: { pendingTransactions: transactionId }
      }
    );
  } catch (error) {
    // Rollback
    await db.orders.updateOne(
      { _id: transactionId },
      { $set: { status: "cancelled" } }
    );
    
    await db.inventory.updateOne(
      { productId: productId },
      { 
        $inc: { reserved: -quantity },
        $pull: { pendingTransactions: transactionId }
      }
    );
  }
}
```"

---

## Part 7: Testing & Validation

### Discussion Point
*"How do you ensure the migration is successful and the new MongoDB system works correctly?"*

**Strong Answer:**
"Comprehensive testing strategy across multiple levels:

**Level 1: Data Integrity Testing**

```javascript
// Validate row count
async function validateCounts() {
  const collections = ['users', 'orders', 'products'];
  
  for (const collection of collections) {
    const sqlCount = await sqlDb.query(
      `SELECT COUNT(*) as count FROM ${collection}`
    );
    const mongoCount = await db[collection].countDocuments();
    
    console.log(`${collection}: SQL=${sqlCount[0].count}, Mongo=${mongoCount}`);
    
    if (sqlCount[0].count !== mongoCount) {
      throw new Error(`Count mismatch in ${collection}`);
    }
  }
}

// Validate data checksums
async function validateDataIntegrity() {
  // Sample random records
  const sampleSize = 10000;
  const sqlSample = await sqlDb.query(
    `SELECT id, user_id, total, MD5(CONCAT(id, user_id, total)) as checksum 
     FROM orders 
     ORDER BY RAND() 
     LIMIT ?`,
    [sampleSize]
  );
  
  for (const row of sqlSample) {
    const mongoDoc = await db.orders.findOne({
      orderNumber: row.id
    });
    
    if (!mongoDoc) {
      console.error(`Missing order: ${row.id}`);
      continue;
    }
    
    // Validate key fields
    const mongoChecksum = md5(
      `${mongoDoc.orderNumber}${mongoDoc.userId}${mongoDoc.total}`
    );
    
    if (row.checksum !== mongoChecksum) {
      console.error(`Data mismatch: ${row.id}`, {
        sql: row,
        mongo: mongoDoc
      });
    }
  }
}

// Validate relationships
async function validateRelationships() {
  // Check orphaned records
  const ordersWithInvalidUsers = await db.orders.aggregate([
    {
      $lookup: {
        from: "users",
        localField: "userId",
        foreignField: "_id",
        as: "user"
      }
    },
    { $match: { user: [] } },  // No matching user
    { $count: "orphanedOrders" }
  ]);
  
  if (orphanedOrders.length > 0) {
    console.error(`Found ${orphanedOrders[0].orphanedOrders} orphaned orders`);
  }
}
```

**Level 2: Performance Testing**

```javascript
// Benchmark common queries
async function performanceTest() {
  const queries = [
    {
      name: "Get user orders",
      fn: () => db.orders.find({ userId: testUserId }).toArray()
    },
    {
      name: "Search products",
      fn: () => db.products.find({
        $text: { $search: "laptop" },
        price: { $lt: 1000 }
      }).toArray()
    },
    {
      name: "Recent orders",
      fn: () => db.orders.find({
        createdAt: { $gte: new Date(Date.now() - 7*24*60*60*1000) }
      }).sort({ createdAt: -1 }).limit(100).toArray()
    }
  ];
  
  for (const query of queries) {
    const iterations = 100;
    const start = Date.now();
    
    for (let i = 0; i < iterations; i++) {
      await query.fn();
    }
    
    const avgTime = (Date.now() - start) / iterations;
    console.log(`${query.name}: ${avgTime.toFixed(2)}ms`);
  }
}

// Load testing
async function loadTest() {
  const concurrency = 50;
  const duration = 60000; // 1 minute
  
  const startTime = Date.now();
  let successCount = 0;
  let errorCount = 0;
  
  const workers = Array(concurrency).fill(0).map(async () => {
    while (Date.now() - startTime < duration) {
      try {
        await db.orders.find({ status: "pending" }).limit(10).toArray();
        successCount++;
      } catch (error) {
        errorCount++;
      }
    }
  });
  
  await Promise.all(workers);
  
  const totalRequests = successCount + errorCount;
  const qps = totalRequests / (duration / 1000);
  
  console.log(`
    Total Requests: ${totalRequests}
    QPS: ${qps.toFixed(2)}
    Success: ${successCount}
    Errors: ${errorCount}
    Error Rate: ${(errorCount/totalRequests*100).toFixed(2)}%
  `);
}
```

**Level 3: Functional Testing**

```javascript
// Test critical workflows
describe('Order Creation Workflow', () => {
  it('should create order and update inventory', async () => {
    const initialStock = await db.products.findOne(
      { _id: productId },
      { projection: { stock: 1 } }
    );
    
    const order = await createOrder({
      userId: testUserId,
      items: [{ productId, quantity: 2 }]
    });
    
    expect(order).toBeDefined();
    expect(order.status).toBe('confirmed');
    
    const updatedStock = await db.products.findOne(
      { _id: productId },
      { projection: { stock: 1 } }
    );
    
    expect(updatedStock.stock).toBe(initialStock.stock - 2);
  });
  
  it('should rollback on insufficient inventory', async () => {
    const product = await db.products.findOne({ _id: productId });
    
    try {
      await createOrder({
        userId: testUserId,
        items: [{ productId, quantity: product.stock + 100 }]
      });
      
      fail('Should have thrown insufficient inventory error');
    } catch (error) {
      expect(error.message).toContain('Insufficient inventory');
      
      // Verify no order was created
      const orders = await db.orders.find({
        userId: testUserId,
        status: 'confirmed'
      }).toArray();
      
      expect(orders).toHaveLength(0);
    }
  });
});

// Test data consistency
describe('Data Consistency', () => {
  it('should maintain referential integrity', async () => {
    const order = await db.orders.findOne({ userId: testUserId });
    const user = await db.users.findOne({ _id: order.userId });
    
    expect(user).toBeDefined();
    expect(user._id).toEqual(order.userId);
  });
  
  it('should have matching embedded data', async () => {
    const order = await db.orders.findOne({});
    const product = await db.products.findOne({
      _id: order.items[0].productId
    });
    
    // SKU should match
    expect(order.items[0].sku).toBe(product.sku);
  });
});
```

**Level 4: Shadow Testing**

```javascript
// Compare MongoDB vs SQL results
async function shadowTest() {
  const testUserId = '12345';
  
  // Query both databases
  const [sqlOrders, mongoOrders] = await Promise.all([
    sqlDb.query('SELECT * FROM orders WHERE user_id = ?', [testUserId]),
    db.orders.find({ userId: ObjectId(testUserId) }).toArray()
  ]);
  
  // Compare results
  if (sqlOrders.length !== mongoOrders.length) {
    logDiscrepancy('count_mismatch', {
      sql: sqlOrders.length,
      mongo: mongoOrders.length
    });
  }
  
  // Compare each order
  for (const sqlOrder of sqlOrders) {
    const mongoOrder = mongoOrders.find(
      o => o.orderNumber === sqlOrder.id
    );
    
    if (!mongoOrder) {
      logDiscrepancy('missing_order', { orderId: sqlOrder.id });
      continue;
    }
    
    if (Math.abs(sqlOrder.total - mongoOrder.total) > 0.01) {
      logDiscrepancy('total_mismatch', {
        orderId: sqlOrder.id,
        sql: sqlOrder.total,
        mongo: mongoOrder.total
      });
    }
  }
}
```

**Level 5: Monitoring & Alerts**

```javascript
// Set up monitoring
async function setupMonitoring() {
  // Alert on slow queries
  db.setProfilingLevel(1, { slowms: 100 });
  
  setInterval(async () => {
    const slowQueries = await db.system.profile.find({
      millis: { $gt: 100 },
      ts: { $gt: new Date(Date.now() - 60000) }
    }).toArray();
    
    if (slowQueries.length > 10) {
      alertTeam('High number of slow queries', {
        count: slowQueries.length,
        queries: slowQueries.slice(0, 5)
      });
    }
  }, 60000);
  
  // Monitor error rates
  let errorCount = 0;
  let requestCount = 0;
  
  app.use((req, res, next) => {
    requestCount++;
    
    res.on('finish', () => {
      if (res.statusCode >= 500) {
        errorCount++;
      }
      
      // Check error rate every 100 requests
      if (requestCount % 100 === 0) {
        const errorRate = errorCount / requestCount;
        if (errorRate > 0.05) {  // 5% threshold
          alertTeam('High error rate', {
            rate: errorRate,
            errors: errorCount,
            total: requestCount
          });
        }
        
        // Reset counters
        errorCount = 0;
        requestCount = 0;
      }
    });
    
    next();
  });
}
```

**Testing Timeline:**

```
Week 1-2: Data integrity testing
  - Validate all records migrated
  - Check relationships
  - Verify data accuracy

Week 3-4: Performance testing
  - Benchmark queries
  - Load testing
  - Identify bottlenecks

Week 5-6: Functional testing
  - Test all workflows
  - Edge cases
  - Error handling

Week 7-8: Shadow testing
  - Run both systems in parallel
  - Compare results
  - Fix discrepancies

Week 9-10: Production monitoring
  - Real traffic testing
  - Performance monitoring
  - User feedback
```"

---

## Additional Topics to Cover

### Security During Migration
*"How do you ensure security during and after the migration?"*

### Rollback Strategy
*"If something goes wrong in production, how do you roll back?"*

### Team Training
*"How do you prepare your development team for working with MongoDB?"*

### Ongoing Maintenance
*"After migration, what ongoing maintenance tasks are different from SQL?"*

---

## Red Flags to Watch For

❌ Doesn't consider query patterns when designing schema
❌ Plans to migrate everything at once (big-bang)
❌ No rollback strategy
❌ Doesn't understand transactions vs single-document atomicity
❌ Over-uses $lookup (treating MongoDB like SQL)
❌ No data validation plan
❌ Doesn't consider team training needs
❌ No performance comparison planned

## Green Flags to Look For

✅ Asks clarifying questions about access patterns
✅ Suggests phased migration approach
✅ Mentions data validation and testing
✅ Understands trade-offs of different schema designs
✅ Has experience with production migrations
✅ Considers operational aspects (monitoring, alerts)
✅ Thinks about team enablement
✅ Discusses both technical and business risks [
    {
      userId: ObjectId("user123"),
      userName: "John Doe",  // Denormalized
      email: "john@example.com",
      role: "admin",
      joinedDate: ISODate("2023-01-15")
    },
    {
      userId: ObjectId("user456"),
      userName: "Jane Smith",
      email: "jane@example.com",
      role: "member",
      joinedDate: ISODate("2023-02-01")
    }
  ]
}
```

**Pros:**
- Fast queries from either direction
- No joins needed

**Cons:**
- Data duplication
- Must update both collections when relationship changes

**When to use:** High read volume, infrequent updates

---

**Approach 2: Reference with Separate Collection**

```javascript
// Users collection
{
  _id: ObjectId("user123"),
  name: "John Doe",
  email: "john@example.com"
}

// Organizations collection
{
  _id: ObjectId("org456"),
  name: "Acme Corp",
  type: "enterprise"
}

// UserOrganizations collection (junction)
{
  _id: ObjectId("..."),
  userId: ObjectId("user123"),
  organizationId: ObjectId("org456"),
  role: "admin",
  joinedDate: ISODate("2023-01-15"),
  permissions: ["read", "write", "delete"]
}

// Query: Get user's organizations
db.userOrganizations.aggregate([
  { $match: { userId: ObjectId("user123") } },
  {
    $lookup: {
      from: "organizations",
      localField: "organizationId",
      foreignField: "_id",
      as: "organization"
    }
  },
  { $unwind: "$organization" }
]);
```

**Pros:**
- No duplication
- Easy to update relationships
- Similar to SQL approach

**Cons:**
- Requires $lookup (slower)
- Multiple collections to maintain

**When to use:** Complex relationship data, frequent updates

---

**Approach 3: Hybrid (Recommended for Most Cases)**

```javascript
// Users collection - embed minimal org data
{
  _id: ObjectId("user123"),
  name: "John Doe",
  email: "john@example.com",
  organizations: [
    {
      orgId: ObjectId("org456"),
      role: "admin"  // Only role here
    },
    {
      orgId: ObjectId("org789"),
      role: "member"
    }
  ]
}

// Organizations collection - embed minimal user data
{
  _id: ObjectId("org456"),
  name: "Acme Corp",
  type: "enterprise",
  settings: { /* ... */ },
  memberCount: 25,  // Cached count
  admins: [ObjectId("user123")],  // Quick admin lookup
  members: