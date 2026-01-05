# Advanced Detailed Prompting Guide for Cursor AI

## The Anatomy of an Excellent Prompt

A high-quality prompt includes:
1. **Context** - Background information about your project
2. **Objective** - What you want to achieve
3. **Technical Constraints** - Tools, frameworks, versions
4. **Requirements** - Functional and non-functional needs
5. **Examples** - Input/output samples if applicable
6. **Quality Standards** - Code style, patterns, best practices

---

## Template: Super Detailed Prompt Structure

```
# CONTEXT
[Provide background about your project, current architecture, and what you're building]

# OBJECTIVE
[Clear statement of what you want to accomplish]

# CURRENT STATE
[Describe existing code/infrastructure if relevant]
- Current implementation: [describe]
- Pain points: [list issues]
- What works well: [list positives]

# TECHNICAL ENVIRONMENT
- Language: [e.g., Python 3.11]
- Framework: [e.g., FastAPI 0.104]
- Database: [e.g., PostgreSQL 15]
- Cloud Platform: [e.g., GCP]
- Other dependencies: [list key libraries]

# REQUIREMENTS

## Functional Requirements
1. [Requirement 1 with acceptance criteria]
2. [Requirement 2 with acceptance criteria]
3. [Requirement 3 with acceptance criteria]

## Non-Functional Requirements
- Performance: [e.g., Handle 1000 requests/sec]
- Security: [e.g., OAuth 2.0, encrypt PII]
- Scalability: [e.g., Support 10M records]
- Maintainability: [e.g., Follow clean code principles]
- Error Handling: [e.g., Graceful degradation]

## Constraints
- Must use: [specific libraries/patterns]
- Must avoid: [antipatterns/deprecated methods]
- API limits: [rate limits, quotas]
- Budget: [cost constraints]

# EXPECTED OUTPUT

## Code Structure
[Describe how you want code organized]
- File structure
- Class/function organization
- Separation of concerns

## Code Quality Standards
- Add comprehensive type hints/annotations
- Include docstrings for all public functions
- Follow [PEP 8/Google Style/Airbnb] style guide
- Add inline comments for complex logic
- Maximum function length: [X lines]
- Maximum cyclomatic complexity: [X]

## Testing Requirements
- Unit test coverage: [X%]
- Include edge cases and error scenarios
- Use [pytest/jest/junit] framework
- Mock external dependencies

## Documentation
- Include README section explaining usage
- Add example code snippets
- Document environment variables
- List prerequisites

# EXAMPLES

## Input Example
```
[Show sample input data/request]
```

## Expected Output
```
[Show expected result/response]
```

## Error Scenarios
```
[Show how errors should be handled]
```

# ADDITIONAL CONTEXT
- Team conventions: [coding standards your team follows]
- Existing patterns: [patterns already used in codebase]
- Future considerations: [upcoming changes to plan for]

# SPECIFIC QUESTIONS TO ADDRESS
1. [Specific technical decision you need help with]
2. [Trade-off you're considering]
3. [Best practice question]

# DELIVERABLES
Please provide:
1. ✅ Complete, production-ready implementation
2. ✅ Unit tests with >80% coverage
3. ✅ Inline documentation and docstrings
4. ✅ Usage examples
5. ✅ Error handling for all edge cases
6. ✅ Performance considerations/optimizations
7. ✅ Security considerations if applicable
8. ✅ Brief explanation of key decisions made
```

---

## Real-World Example: Salesforce to BigQuery Data Validator

```
# CONTEXT
I'm building a Salesforce to BigQuery ETL pipeline. We extract data using Salesforce Bulk API 2.0, transform it, and load into BigQuery. The pipeline runs daily via Apache Airflow. We need a comprehensive data validation module that runs after extraction and before loading to ensure data quality.

# OBJECTIVE
Create a reusable data validation framework that validates Salesforce extracted data against predefined rules before loading to BigQuery. The validator should catch data quality issues early and prevent bad data from entering our data warehouse.

# CURRENT STATE
- Current implementation: Basic row count validation only
- Pain points: 
  - Bad data entering BigQuery (NULLs in required fields)
  - Schema mismatches causing load failures
  - Referential integrity violations
  - No audit trail of validation failures
- What works well: Basic pipeline structure is solid

# TECHNICAL ENVIRONMENT
- Language: Python 3.11
- Framework: Apache Airflow 2.7
- Data Processing: Pandas 2.1
- BigQuery: google-cloud-bigquery 3.13
- Storage: Google Cloud Storage
- Logging: Python logging + Cloud Logging

# REQUIREMENTS

## Functional Requirements
1. Validate required fields are not NULL/empty
   - Accept configuration of required fields per object
   - Report all violations, not just first failure
   
2. Validate data types match expected schema
   - String, Integer, Decimal, Date, DateTime, Boolean
   - Handle Salesforce-specific types (Id, Email, Phone)
   
3. Validate referential integrity
   - Check foreign key references exist (e.g., Contact.AccountId exists in Account)
   - Handle both intra-object and cross-object references
   
4. Validate business rules
   - Custom validation rules (e.g., "Opportunity Amount must be positive")
   - Regex patterns for formats (email, phone)
   - Value ranges (dates, amounts)
   
5. Validate record counts
   - Compare against expected ranges
   - Flag anomalies (>20% deviation from average)
   
6. Generate detailed validation report
   - Summary: total records, passed, failed
   - Details: which records failed, which rules violated
   - Store report in BigQuery audit table
   
## Non-Functional Requirements
- Performance: Validate 1M records in <5 minutes
- Memory Efficiency: Handle large files via chunking (100K rows/chunk)
- Extensibility: Easy to add new validation rules
- Configurability: Rules defined in YAML/JSON config
- Observability: Log progress every 10K records
- Error Handling: Continue validation even if some rules fail

## Constraints
- Must use: Pandas for data manipulation
- Must avoid: Loading entire dataset in memory
- Must integrate with: Existing Airflow DAG structure
- File format: CSV files in GCS

# EXPECTED OUTPUT

## Code Structure
```
validators/
├── __init__.py
├── base_validator.py          # Abstract base class
├── field_validator.py          # NULL, type, format checks
├── reference_validator.py      # Foreign key checks
├── business_rule_validator.py  # Custom rules
├── validator_config.py         # Config loader
├── validation_report.py        # Report generator
└── exceptions.py               # Custom exceptions

config/
└── validation_rules.yaml       # Validation rule definitions

tests/
└── test_validators.py
```

## Code Quality Standards
- Add comprehensive type hints using typing module
- Include docstrings (Google style) for all classes and public methods
- Follow PEP 8 style guide
- Add inline comments for complex validation logic
- Maximum function length: 50 lines
- Use dataclasses for configuration objects

## Testing Requirements
- Unit test coverage: 85%+
- Test cases:
  - Happy path: all validations pass
  - Each validation rule failing individually
  - Multiple validation failures
  - Large dataset handling (1M records)
  - Invalid configuration
- Use pytest with fixtures
- Mock GCS and BigQuery interactions

## Documentation
- README section with:
  - Quick start guide
  - Configuration format explanation
  - How to add custom validators
  - Example validation rules
- Docstring for every validator class explaining its purpose

# EXAMPLES

## Configuration File (validation_rules.yaml)
```yaml
Account:
  required_fields:
    - Id
    - Name
  type_validations:
    Id:
      type: string
      pattern: "^[a-zA-Z0-9]{18}$"
    AnnualRevenue:
      type: decimal
      min: 0
  business_rules:
    - rule: "AnnualRevenue must be positive if provided"
      condition: "AnnualRevenue IS NULL OR AnnualRevenue > 0"

Contact:
  required_fields:
    - Id
    - AccountId
    - Email
  reference_validations:
    AccountId:
      references: "Account.Id"
  type_validations:
    Email:
      type: string
      pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
```

## Usage Example
```python
from validators import DataValidator

# Initialize validator
validator = DataValidator(
    config_path="config/validation_rules.yaml",
    object_name="Account"
)

# Validate data
result = validator.validate_file(
    gcs_path="gs://bucket/salesforce/Account_20240115.csv",
    chunk_size=100000
)

# Check results
if result.is_valid:
    print(f"✓ Validation passed: {result.total_records} records")
else:
    print(f"✗ Validation failed: {result.failed_records} failures")
    result.save_to_bigquery("project.dataset.validation_log")
```

## Expected Validation Report
```python
{
    "object_name": "Account",
    "file_path": "gs://bucket/salesforce/Account_20240115.csv",
    "timestamp": "2024-01-15T10:30:00Z",
    "total_records": 50000,
    "valid_records": 49850,
    "invalid_records": 150,
    "validation_duration_seconds": 45.2,
    "failures": [
        {
            "record_id": "001XX000003DHP0",
            "rule": "required_field",
            "field": "Name",
            "message": "Required field 'Name' is NULL or empty"
        },
        {
            "record_id": "001XX000003DHP1",
            "rule": "type_validation",
            "field": "AnnualRevenue",
            "message": "Expected decimal, got string: 'N/A'"
        }
    ],
    "summary_by_rule": {
        "required_field": 80,
        "type_validation": 45,
        "reference_validation": 25
    }
}
```

# ADDITIONAL CONTEXT
- Team conventions: 
  - Use dataclasses over dicts for structured data
  - Prefer composition over inheritance
  - Follow "fail fast" principle for config errors
  
- Existing patterns:
  - We use Abstract Base Classes for plugins
  - Configuration always loaded from YAML
  - All GCS operations go through custom GCSClient wrapper
  
- Future considerations:
  - May add real-time validation for streaming inserts
  - Planning to support validation of Parquet files
  - Will need to scale to 10M+ records in future

# SPECIFIC QUESTIONS TO ADDRESS
1. Should validation continue after first rule failure per record, or fail fast?
   - I'm thinking continue to collect all failures for better reporting
   
2. How to handle validation of cross-object references efficiently?
   - Load referenced data into memory? Query as needed? Pre-build lookup dict?
   
3. Best way to make validators extensible for custom business rules?
   - Thinking: allow Python functions in config, or SQL-like expressions?

# DELIVERABLES
Please provide:
1. ✅ Complete validator framework with base classes and implementations
2. ✅ Unit tests covering all validation types (>85% coverage)
3. ✅ Sample validation_rules.yaml with comments
4. ✅ README section explaining architecture and usage
5. ✅ Error handling for malformed configs, file read errors
6. ✅ Performance optimizations (chunking, memory management)
7. ✅ Integration example with Airflow (PythonOperator snippet)
8. ✅ Explanation of design decisions (e.g., why certain patterns chosen)
9. ✅ BigQuery schema for validation_log table

# SUCCESS CRITERIA
The implementation is successful if:
- ✓ Can validate 1M records in under 5 minutes
- ✓ Memory usage stays under 2GB for large files
- ✓ Easy to add new validation rules via config only
- ✓ Produces actionable validation reports
- ✓ Integrates seamlessly with our Airflow DAG
- ✓ All tests pass with >85% coverage
```

---

## Example: Detailed Debugging Prompt

```
# CONTEXT
I'm working on a Salesforce to BigQuery ETL pipeline. The pipeline extracts data using Salesforce Bulk API, transforms it with Pandas, and loads to BigQuery.

# THE PROBLEM
The pipeline is failing intermittently with BigQuery load errors. About 1 in 5 runs fails with this error:

```
google.cloud.exceptions.BadRequest: 400 POST https://bigquery.googleapis.com/bigquery/v2/projects/my-project/jobs:
Error while reading data, error message: Could not parse '2024-01-15T25:00:00Z' as TIMESTAMP for field created_date (line 1234)
```

# CURRENT IMPLEMENTATION
```python
# Extract from Salesforce
def extract_salesforce_data(object_name: str) -> pd.DataFrame:
    # ... Bulk API extraction code ...
    df = pd.read_csv(f"gs://bucket/{object_name}.csv")
    return df

# Transform timestamps
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_cols = ['CreatedDate', 'LastModifiedDate', 'CloseDate']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

# Load to BigQuery
def load_to_bigquery(df: pd.DataFrame, table_id: str):
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
```

# WHAT I'VE TRIED
1. Added error handling around pd.to_datetime() - still fails
2. Checked Salesforce data - looks valid in Salesforce UI
3. Tried setting `errors='coerce'` in pd.to_datetime() - converts to NaT but still fails
4. Pipeline succeeds on small test datasets (<1000 records)

# ENVIRONMENT
- Python: 3.11
- Pandas: 2.1.0
- google-cloud-bigquery: 3.13.0
- Salesforce API: REST API v58.0
- BigQuery schema: created_date TIMESTAMP

# ADDITIONAL OBSERVATIONS
- Error only happens on large datasets (>10K records)
- Error message shows impossible timestamp: "T25:00:00" (hour 25 doesn't exist)
- Line number in error varies each run
- Same Salesforce object sometimes loads successfully

# WHAT I NEED
1. Root cause analysis:
   - Why are invalid timestamps appearing?
   - Why is it intermittent?
   - Why does pd.to_datetime() not catch this?

2. Robust solution:
   - Validate and clean timestamps before BigQuery load
   - Handle edge cases and invalid data
   - Log problematic records for investigation
   - Ensure pipeline doesn't fail on bad data

3. Diagnostic code:
   - Add validation step to detect bad timestamps
   - Create data quality report
   - Sample and log invalid records

4. Prevention:
   - How to prevent this at source (Salesforce extraction)?
   - Best practices for timestamp handling in ETL?

# SPECIFIC QUESTIONS
1. Could Salesforce API be returning corrupted data occasionally?
2. Is this a timezone conversion issue?
3. Should I validate timestamps before or after Pandas conversion?
4. What's the best way to handle invalid timestamps - skip records, null them out, or fail pipeline?

# DELIVERABLES NEEDED
1. ✅ Explanation of why this is happening
2. ✅ Code to detect and log all invalid timestamps in dataset
3. ✅ Robust timestamp validation and cleaning function
4. ✅ Updated ETL code with validation step
5. ✅ Error handling that prevents pipeline failure
6. ✅ Data quality check that runs before BigQuery load
7. ✅ Test cases for various invalid timestamp formats
8. ✅ Recommendations for preventing this issue
```

---

## Tips for Effective Detailed Prompts

### DO:
- ✅ Provide actual error messages and stack traces
- ✅ Share relevant code snippets (not entire files)
- ✅ Specify versions of tools and libraries
- ✅ Describe what you've already tried
- ✅ Include expected vs actual behavior
- ✅ Ask specific questions
- ✅ Define success criteria

### DON'T:
- ❌ Say "it doesn't work" without details
- ❌ Dump entire codebases without context
- ❌ Ask vague questions like "how do I make it better?"
- ❌ Omit technical environment details
- ❌ Forget to mention constraints
- ❌ Skip examples of desired output

### Advanced Techniques:
1. **Use structured sections** - Makes prompts scannable
2. **Provide before/after examples** - Shows desired transformation
3. **List explicit requirements** - Numbered lists are clear
4. **Include success criteria** - Defines "done"
5. **Ask follow-up questions** - Guides the solution
6. **Specify what NOT to do** - Avoids unwanted approaches

---

## Quick Reference: Prompt Checklist

Before submitting your prompt, verify:

- [ ] Context: Did I explain what I'm building?
- [ ] Problem: Is the objective crystal clear?
- [ ] Environment: Did I list languages, versions, tools?
- [ ] Requirements: Are functional & non-functional needs specified?
- [ ] Examples: Did I show input/output samples?
- [ ] Constraints: Did I mention limitations?
- [ ] Current state: Did I share existing code if relevant?
- [ ] Quality standards: Did I specify code style preferences?
- [ ] Deliverables: Did I list what I want to receive?
- [ ] Success criteria: How will I know it's correct?