# Salesforce to BigQuery Integration - Cursor AI Prompts

## Data Extraction & API Integration

### Salesforce REST API Client
```
Create a Salesforce REST API client for data extraction:

Requirements:
- Implement OAuth 2.0 authentication flow
- Handle token refresh automatically
- Support SOQL query execution
- Add bulk API support for large datasets
- Include rate limiting (API call limits)
- Add retry logic with exponential backoff
- Handle Salesforce API errors gracefully
- Log all API interactions
- Use TypeScript with proper types

Objects to extract: [Account, Contact, Opportunity, Custom_Object__c, etc.]
```

### Salesforce Bulk API Implementation
```
Implement Salesforce Bulk API 2.0 for extracting large datasets:

Objects: [specify objects]
Record volume: [estimated records per object]

Requirements:
- Create bulk query jobs
- Poll for job completion
- Download result CSV files
- Handle job failures and retries
- Support incremental extracts (LastModifiedDate filter)
- Implement batch processing
- Add progress tracking
- Memory-efficient streaming for large files
```

### Incremental Data Sync
```
Design an incremental sync strategy for Salesforce to BigQuery:

Objects to sync: [list objects]
Sync frequency: [hourly/daily/real-time]

Requirements:
- Track last sync timestamp per object
- Query only changed records (LastModifiedDate, SystemModstamp)
- Handle deleted records (IsDeleted flag)
- Store watermarks in [location: BigQuery table/Cloud Storage/database]
- Implement upsert logic (insert or update)
- Add data validation before load
- Create audit trail of sync operations
```

## Data Transformation & Schema Mapping

### Schema Mapper
```
Create schema mapping from Salesforce objects to BigQuery tables:

Salesforce Objects:
- [Object1]: [fields]
- [Object2]: [fields]

Requirements:
- Map Salesforce data types to BigQuery types
- Handle picklist values (convert to STRING or create dimension table)
- Transform datetime fields to BigQuery TIMESTAMP
- Handle multi-picklist (convert to ARRAY or separate table)
- Map lookup relationships to foreign keys
- Create flattened structure for parent-child relationships
- Document all transformations
- Generate BigQuery DDL statements
```

### Data Transformation Pipeline
```
Build data transformation pipeline for Salesforce data:

Source: [Salesforce objects]
Destination: [BigQuery dataset.table]

Transformations needed:
- Clean and standardize field names (remove __c, spaces)
- Parse and validate data types
- Handle NULL values and empty strings
- Convert compound fields (Address, Name)
- Denormalize lookup relationships
- Apply business rules: [specify]
- Add metadata columns (extracted_at, source_system)
- Validate data quality rules

Use: [Python/Apache Beam/Dataflow/dbt]
```

### Relationship Handler
```
Handle Salesforce object relationships in BigQuery:

Relationships:
- Account → Contacts (1:Many)
- Opportunity → Account (Many:1)
- [Custom relationships]

Requirements:
- Create junction tables for Many:Many
- Preserve foreign key relationships
- Option 1: Denormalize (flatten) relationships
- Option 2: Maintain normalized structure
- Add relationship metadata table
- Handle lookup and master-detail relationships
- Create views for common joins
```

## BigQuery Loading & Optimization

### BigQuery Load Strategy
```
Design BigQuery data loading strategy:

Load method: [Batch/Streaming]
Data volume: [records per day]
Tables: [list tables]

Requirements:
- Use appropriate load method (batch vs streaming)
- Implement partitioning strategy (by date/timestamp)
- Add clustering on frequently queried columns
- Choose merge vs append strategy
- Handle schema evolution
- Set up staging tables
- Implement data quality checks pre-load
- Add load metadata tracking
```

### Partitioning & Clustering
```
Optimize BigQuery tables with partitioning and clustering:

Tables: [list tables with query patterns]

For each table:
- Analyze query patterns
- Recommend partition column (usually date/timestamp)
- Recommend clustering columns (up to 4)
- Estimate cost savings
- Generate CREATE TABLE statements
- Handle partition maintenance
- Document partition strategy
```

### BigQuery Merge/Upsert Implementation
```
Implement MERGE statement for upserting Salesforce data:

Target table: [dataset.table]
Source: [staging table or temp table]
Primary key: [Id]

Requirements:
- Use MERGE statement for upsert
- Match on Salesforce Id
- Update changed records
- Insert new records
- Optionally handle soft deletes
- Add last_modified timestamp
- Optimize for performance (partition pruning)
- Handle large dataset efficiently
- Add error handling and validation
```

## Orchestration & Workflow

### Airflow DAG for ETL Pipeline
```
Create Apache Airflow DAG for Salesforce to BigQuery ETL:

Objects: [list objects]
Schedule: [cron expression]

DAG structure:
1. Extract from Salesforce (Bulk API)
2. Validate extracted data
3. Transform data
4. Load to BigQuery staging
5. Run data quality checks
6. Merge to production tables
7. Send success/failure notification

Requirements:
- Add task dependencies
- Implement error handling and retries
- Add SLA monitoring
- Create task groups for multiple objects
- Include data quality sensors
- Add alerting (email/Slack)
- Log execution metrics
```

### Cloud Functions/Lambda Orchestration
```
Create serverless orchestration for real-time sync:

Trigger: [Salesforce Platform Events/Change Data Capture/Webhook]
Target: BigQuery [dataset.table]

Requirements:
- Set up event listener
- Parse Salesforce event payload
- Transform data
- Stream to BigQuery
- Handle failures with retry queue (Pub/Sub)
- Add dead letter queue
- Implement idempotency
- Monitor latency
- Scale automatically
```

### Error Handling & Recovery
```
Implement comprehensive error handling:

Error scenarios:
- API rate limits exceeded
- Authentication failures
- Network timeouts
- Data validation failures
- BigQuery load failures
- Schema mismatches

Requirements:
- Categorize errors (transient vs permanent)
- Implement retry logic with backoff
- Create error logging table in BigQuery
- Send alerts for critical failures
- Implement circuit breaker pattern
- Add manual recovery process
- Create error dashboard
```

## Data Quality & Monitoring

### Data Quality Checks
```
Implement data quality validation:

Checks needed:
- Record count validation (source vs destination)
- Schema validation
- NULL value checks for required fields
- Data type validation
- Referential integrity checks
- Business rule validation: [specify]
- Duplicate detection
- Outlier detection

Requirements:
- Run checks post-extraction and post-load
- Store results in BigQuery audit table
- Fail pipeline on critical issues
- Create data quality dashboard
- Send alerts on quality thresholds
```

### Monitoring Dashboard
```
Create monitoring dashboard for ETL pipeline:

Metrics to track:
- Records extracted per object
- Records loaded to BigQuery
- Pipeline execution time
- API calls consumed
- Error rates
- Data freshness (lag time)
- Cost per run
- Storage growth

Requirements:
- Use [Looker/Data Studio/Tableau/Grafana]
- Pull data from BigQuery audit tables
- Add alerting on anomalies
- Show historical trends
- Include drill-down capability
```

## Security & Compliance

### Salesforce Authentication Setup
```
Implement secure Salesforce authentication:

Requirements:
- Use OAuth 2.0 with JWT bearer flow
- Store credentials securely (Secret Manager/Key Vault)
- Implement least privilege (create integration user)
- Rotate tokens regularly
- Add IP restrictions if needed
- Log all authentication attempts
- Handle token expiration gracefully
- Document security setup
```

### PII Data Handling
```
Implement PII data protection:

PII fields: [Email, Phone, SSN, etc.]

Requirements:
- Identify all PII fields
- Implement field-level encryption for sensitive data
- Use BigQuery column-level security
- Add data masking for non-prod environments
- Implement row-level security if needed
- Create audit trail for PII access
- Document data retention policies
- Ensure GDPR/CCPA compliance
```

## Specific Implementation Patterns

### Full vs Incremental Load Logic
```
Implement logic to choose full vs incremental load:

Objects: [list objects]

Requirements:
- Check if table exists in BigQuery
- If not exists → full load
- If exists → check last sync timestamp
- Query Salesforce with date filter
- Handle edge cases (schema changes, data backfill)
- Add override parameter for manual full refresh
- Log load type for each run
```

### Schema Change Detection
```
Implement Salesforce schema change detection:

Requirements:
- Fetch current Salesforce object metadata
- Compare with stored schema (previous run)
- Detect new fields, removed fields, type changes
- Update BigQuery schema automatically or alert
- Handle breaking vs non-breaking changes
- Version schema changes
- Create migration scripts if needed
- Test backward compatibility
```

### Cost Optimization
```
Optimize pipeline costs:

Current costs: [breakdown if known]

Optimization strategies:
- Use Bulk API instead of REST API
- Partition BigQuery tables by date
- Use BigQuery streaming inserts only when needed
- Compress data in Cloud Storage before load
- Expire old partitions
- Use slots reservation for predictable workloads
- Optimize query performance (avoid SELECT *)
- Monitor and alert on cost spikes
```