# LangChain Document Loaders Examples

This repository demonstrates how to use various document loaders in [LangChain](https://python.langchain.com/) and how to integrate them with LLMs (OpenAI). Examples progress from basic to advanced usage.

## Table of Contents

1. [Basic Text Loader](#1-basic-text-loader)
2. [PDF Loader](#2-pdf-loader)
3. [Web Loader](#3-web-loader)
4. [Integration with LLM](#4-integration-with-llm)
5. [Advanced: Multi-Source Loader](#5-advanced-multi-source-loader)

---

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your OpenAI API key:**
   - Copy `.env.example` to `.env` and add your OpenAI API key.

---

## 1. Basic Text Loader
- **File:** `1_basic_text_loader.py`
- **Description:** Loads a plain text file as a document and prints its content.
- **How to run:**
  ```bash
  python 1_basic_text_loader.py
  ```

## 2. PDF Loader
- **File:** `2_pdf_loader.py`
- **Description:** Loads a PDF file as documents (one per page) and prints their content.
- **How to run:**
  ```bash
  python 2_pdf_loader.py
  ```
  - Requires `sample.pdf` in the same directory.

## 3. Web Loader
- **File:** `3_web_loader.py`
- **Description:** Loads content from a web page as a document and prints the first 500 characters.
- **How to run:**
  ```bash
  python 3_web_loader.py
  ```

## 4. Integration with LLM
- **File:** `4_integration_with_llm.py`
- **Description:** Loads a document and uses OpenAI's LLM to answer a question about its content.
- **How to run:**
  ```bash
  python 4_integration_with_llm.py
  ```
  - Requires OpenAI API key in `.env`.

## 5. Advanced: Multi-Source Loader
- **File:** `5_advanced_multi_source_loader.py`
- **Description:** Loads documents from text, PDF, and web sources, combines them, and uses an LLM to answer a question about all sources.
- **How to run:**
  ```bash
  python 5_advanced_multi_source_loader.py
  ```
  - Handles missing files/sources gracefully.

---

## Notes
- For PDF examples, add your own `sample.pdf` file.
- For LLM examples, ensure your OpenAI API key is set in `.env`.
- All scripts are well-commented for learning purposes. 