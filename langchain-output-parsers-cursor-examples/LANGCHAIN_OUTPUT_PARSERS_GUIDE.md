# LangChain Output Parsers: A Practical Guide

> **Unlock the power of structured LLM output with LangChain's output parsers!**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Why Use Output Parsers?](#why-use-output-parsers)
3. [Parser Types Overview](#parser-types-overview)
4. [Step-by-Step Examples](#step-by-step-examples)
    - [StrOutputParser](#stroutputparser)
    - [CommaSeparatedListOutputParser](#commaseparatedlistoutputparser)
    - [JsonOutputParser](#jsonoutputparser)
    - [PydanticOutputParser](#pydanticoutputparser)
    - [StructuredOutputParser](#structuredoutputparser)
    - [Custom Output Parsers](#custom-output-parsers)
    - [Table/CSV Parsing](#tablecsv-parsing)
    - [Multimodal/Text+Image Parsing](#multimodaltextimage-parsing)
5. [Best Practices](#best-practices)
6. [Real-World Use Cases](#real-world-use-cases)
7. [References & Further Reading](#references--further-reading)

---

## Introduction

LangChain output parsers transform raw LLM output into structured, reliable Python data. This guide covers all major parser types, with code, explanations, and real-world scenarios.

---

## Why Use Output Parsers?
- **Reliability:** Get predictable, structured data from LLMs.
- **Validation:** Catch errors early with schema validation.
- **Productivity:** Automate extraction for analytics, dashboards, and apps.

---

## Parser Types Overview

| Parser Type                        | Use Case Example                        | Example File                                 |
|------------------------------------|-----------------------------------------|----------------------------------------------|
| `StrOutputParser`                  | Extract plain text or summaries         | [01-output-parser-basic.py](01-output-parser-basic.py), [log_summary_str_parser.py](log_summary_str_parser.py) |
| `CommaSeparatedListOutputParser`   | Extract lists from comma-separated text | [02-output-parser-list.py](02-output-parser-list.py), [log_summary_issues_list_parser.py](log_summary_issues_list_parser.py) |
| `JsonOutputParser`                 | Parse output as JSON objects/arrays     | [03-output-parser-dict.py](03-output-parser-dict.py), [movie_review_json_parser.py](movie_review_json_parser.py) |
| `PydanticOutputParser`             | Parse into validated Pydantic models    | [04-output-parser-pydantic.py](04-output-parser-pydantic.py), [movie_review_output_parser.py](movie_review_output_parser.py) |
| `StructuredOutputParser`           | Parse into structured objects (Pydantic)| [11-output-parser-structuredoutput-basic.py](11-output-parser-structuredoutput-basic.py) |
| `Custom Output Parsers`            | Advanced/regex/custom logic             | [05-output-parser-custom.py](05-output-parser-custom.py), [10-output-parser-structured-summary.py](10-output-parser-structured-summary.py) |
| Table/CSV Parsing                  | Parse CSV/tabular output                | [06-output-parser-table.py](06-output-parser-table.py), [15-output-parser-financial-table.py](15-output-parser-financial-table.py) |
| Multimodal/Text+Image Parsing      | Extract text and image URLs             | [08-output-parser-multimodal.py](08-output-parser-multimodal.py) |

---

## Step-by-Step Examples

### StrOutputParser
**Extract plain text or summaries.**
```python
from langchain.output_parsers import StrOutputParser
# ... setup model and prompt ...
parser = StrOutputParser()
result = chain | parser
```
See: [01-output-parser-basic.py](01-output-parser-basic.py)

---

### CommaSeparatedListOutputParser
**Extract lists from comma-separated output.**
```python
from langchain.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
result = chain | parser
```
See: [02-output-parser-list.py](02-output-parser-list.py)

---

### JsonOutputParser
**Parse output as JSON objects or arrays.**
```python
from langchain.output_parsers import JsonOutputParser
parser = JsonOutputParser()
result = chain | parser
```
See: [movie_review_json_parser.py](movie_review_json_parser.py)

---

### PydanticOutputParser
**Parse output into validated Pydantic models.**
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
class MySchema(BaseModel):
    ...
parser = PydanticOutputParser(pydantic_object=MySchema)
result = chain | parser
```
See: [movie_review_output_parser.py](movie_review_output_parser.py)

---

### StructuredOutputParser
**Parse output into structured objects using Pydantic schemas.**
```python
from langchain.output_parsers import StructuredOutputParser
parser = StructuredOutputParser.from_schema(MySchema)
result = chain | parser
```
See: [11-output-parser-structuredoutput-basic.py](11-output-parser-structuredoutput-basic.py)

---

### Custom Output Parsers
**For advanced, custom, or regex-based parsing.**
```python
from langchain_core.output_parsers import BaseOutputParser
class MyCustomParser(BaseOutputParser):
    def parse(self, text: str):
        # custom logic
        return ...
parser = MyCustomParser()
result = chain | parser
```
See: [05-output-parser-custom.py](05-output-parser-custom.py)

---

### Table/CSV Parsing
**Parse CSV/tabular output into lists of dicts.**
```python
import csv, io
from langchain.output_parsers import StrOutputParser
class TableCSVParser(StrOutputParser):
    def parse(self, text: str):
        reader = csv.DictReader(io.StringIO(text.strip()))
        return [row for row in reader]
parser = TableCSVParser()
result = chain | parser
```
See: [06-output-parser-table.py](06-output-parser-table.py)

---

### Multimodal/Text+Image Parsing
**Extract both text and image URLs or other multimodal content.**
```python
import re
from langchain.output_parsers import StrOutputParser
class TextImageParser(StrOutputParser):
    def parse(self, text: str):
        url = ... # regex extract
        return {"description": ..., "image_url": url}
parser = TextImageParser()
result = chain | parser
```
See: [08-output-parser-multimodal.py](08-output-parser-multimodal.py)

---

## Best Practices
- **Prompt clearly:** Instruct the LLM to output in the expected format.
- **Validate output:** Use Pydantic or StructuredOutputParser for critical data.
- **Handle errors:** Add try/except or fallback logic for robustness.
- **Test with real data:** LLMs may hallucinate—always test with real-world samples.

---

## Real-World Use Cases
- **Movie review analytics:** [movie_review_output_parser.py](movie_review_output_parser.py), [movie_review_json_parser.py](movie_review_json_parser.py)
- **Application log summarization:** [log_summary_output_parser.py](log_summary_output_parser.py), [log_summary_str_parser.py](log_summary_str_parser.py), [log_summary_issues_list_parser.py](log_summary_issues_list_parser.py)
- **Customer support chat analysis:** [13-output-parser-customer-support-summary.py](13-output-parser-customer-support-summary.py)
- **E-commerce order extraction:** [14-output-parser-ecommerce-order-json.py](14-output-parser-ecommerce-order-json.py)
- **Financial report parsing:** [15-output-parser-financial-table.py](15-output-parser-financial-table.py)
- **Healthcare patient summaries:** [16-output-parser-healthcare-patient-summary.py](16-output-parser-healthcare-patient-summary.py)
- **Travel itinerary extraction:** [17-output-parser-travel-itinerary.py](17-output-parser-travel-itinerary.py)

---

## References & Further Reading
- [LangChain Output Parsers Documentation](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

---

> **Tip:** Mix and match parser types for complex workflows. For example, use a custom parser to pre-process, then a Pydantic parser for validation! 