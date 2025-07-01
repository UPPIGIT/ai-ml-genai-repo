# LangChain Document Loaders: From Basic to Advanced
# Complete guide with practical examples and detailed comments

from langchain_community.document_loaders import (
    TextLoader, CSVLoader, PyPDFLoader, UnstructuredHTMLLoader,
    DirectoryLoader, WebBaseLoader, GitHubIssuesLoader,
    NotionDirectoryLoader, SlackDirectoryLoader, YoutubeLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from pathlib import Path

# =============================================================================
# BASIC EXAMPLES - Getting Started
# =============================================================================

def basic_text_loader():
    """
    Basic text file loading - simplest way to start
    Use this for: Plain text files, logs, simple documents
    """
    # Step 1: Create a text loader instance
    loader = TextLoader("example.txt", encoding="utf-8")
    
    # Step 2: Load the document
    documents = loader.load()
    
    # Step 3: Examine what we got
    print(f"Loaded {len(documents)} document(s)")
    print(f"Content preview: {documents[0].page_content[:200]}...")
    print(f"Metadata: {documents[0].metadata}")
    
    return documents

def basic_csv_loader():
    """
    Loading CSV files with automatic column detection
    Use this for: Structured data, spreadsheets, database exports
    """
    # Step 1: Basic CSV loading
    loader = CSVLoader(
        file_path="data.csv",
        encoding="utf-8"
    )
    
    # Step 2: Load documents (each row becomes a document)
    documents = loader.load()
    
    # Step 3: Inspect the results
    print(f"Loaded {len(documents)} rows as documents")
    print(f"First row content: {documents[0].page_content}")
    
    return documents

def basic_pdf_loader():
    """
    Simple PDF loading - one document per page
    Use this for: Research papers, reports, books
    """
    # Step 1: Create PDF loader
    loader = PyPDFLoader("document.pdf")
    
    # Step 2: Load all pages
    pages = loader.load()
    
    # Step 3: Process the results
    print(f"PDF has {len(pages)} pages")
    for i, page in enumerate(pages[:3]):  # Show first 3 pages
        print(f"Page {i+1}: {page.page_content[:100]}...")
        print(f"Page {i+1} metadata: {page.metadata}")
    
    return pages

# =============================================================================
# INTERMEDIATE EXAMPLES - More Control and Features
# =============================================================================

def intermediate_directory_loader():
    """
    Load multiple files from a directory with filtering
    Use this for: Processing entire folders, batch operations
    """
    # Step 1: Set up directory loader with file type filtering
    loader = DirectoryLoader(
        path="./documents",           # Directory path
        glob="**/*.txt",             # Pattern to match files
        loader_cls=TextLoader,       # Loader class for each file
        loader_kwargs={              # Arguments for individual loaders
            "encoding": "utf-8"
        },
        show_progress=True,          # Show loading progress
        use_multithreading=True      # Speed up with multiple threads
    )
    
    # Step 2: Load all matching files
    documents = loader.load()
    
    # Step 3: Organize results
    file_count = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        file_count[source] = file_count.get(source, 0) + 1
    
    print(f"Loaded {len(documents)} documents from {len(file_count)} files")
    for file, count in file_count.items():
        print(f"{file}: {count} chunks")
    
    return documents

def intermediate_web_loader():
    """
    Load content from web pages with error handling
    Use this for: Scraping articles, documentation, blogs
    """
    urls = [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
    
    # Step 1: Create web loader with configuration
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={                  # BeautifulSoup arguments
            "features": "html.parser",
            "parse_only": None       # Parse entire document
        },
        header_template={            # Custom headers for requests
            "User-Agent": "Mozilla/5.0 (compatible; LangChain)"
        }
    )
    
    # Step 2: Load with error handling
    try:
        documents = loader.load()
        
        # Step 3: Process successful loads
        for doc in documents:
            # Clean up the content
            content = doc.page_content.strip()
            url = doc.metadata.get('source', 'Unknown URL')
            print(f"Loaded {len(content)} characters from {url}")
            
    except Exception as e:
        print(f"Error loading web content: {e}")
        documents = []
    
    return documents

def intermediate_csv_with_customization():
    """
    Advanced CSV loading with custom column handling
    Use this for: Complex CSV files, data preprocessing
    """
    # Step 1: Load CSV with specific column as content
    loader = CSVLoader(
        file_path="products.csv",
        csv_args={                   # Arguments passed to csv.DictReader
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": None       # Use first row as headers
        },
        content_columns=["description", "features"],  # Columns to use as content
        metadata_columns=["id", "category", "price"], # Columns to keep as metadata
        source_column="product_name"  # Column to use as document source
    )
    
    # Step 2: Load and examine
    documents = loader.load()
    
    # Step 3: Show structure
    for i, doc in enumerate(documents[:3]):
        print(f"Document {i+1}:")
        print(f"  Content: {doc.page_content[:150]}...")
        print(f"  Metadata: {doc.metadata}")
        print()
    
    return documents

# =============================================================================
# ADVANCED EXAMPLES - Complex Scenarios and Integration
# =============================================================================

def advanced_github_loader():
    """
    Load GitHub issues with filtering and authentication
    Use this for: Analyzing project issues, customer feedback
    """
    # Step 1: Set up GitHub loader (requires GitHub token)
    loader = GitHubIssuesLoader(
        repo="langchain-ai/langchain",  # Repository
        access_token=os.getenv("GITHUB_TOKEN"),  # Authentication
        creator="user123",              # Filter by issue creator
        state="open",                   # open, closed, or all
        sort="created",                 # created, updated, comments
        direction="desc",               # asc or desc
        since="2023-01-01T00:00:00Z"   # Issues since this date
    )
    
    # Step 2: Load issues with error handling
    try:
        issues = loader.load()
        
        # Step 3: Process and categorize
        categories = {"bug": [], "feature": [], "other": []}
        
        for issue in issues:
            title = issue.metadata.get("title", "").lower()
            if "bug" in title or "error" in title:
                categories["bug"].append(issue)
            elif "feature" in title or "enhancement" in title:
                categories["feature"].append(issue)
            else:
                categories["other"].append(issue)
        
        # Step 4: Report results
        for category, items in categories.items():
            print(f"{category.title()} issues: {len(items)}")
            
    except Exception as e:
        print(f"Error loading GitHub issues: {e}")
        issues = []
    
    return issues

def advanced_notion_loader():
    """
    Load Notion pages with hierarchical structure
    Use this for: Knowledge bases, documentation, team wikis
    """
    # Step 1: Set up Notion loader (requires integration token)
    loader = NotionDirectoryLoader(
        notion_integration_token=os.getenv("NOTION_TOKEN"),
        database_id="your-database-id",
        request_timeout_sec=30
    )
    
    # Step 2: Load with structure preservation
    try:
        pages = loader.load()
        
        # Step 3: Organize by page type or properties
        organized_pages = {}
        
        for page in pages:
            # Extract page properties from metadata
            page_type = page.metadata.get("type", "unknown")
            if page_type not in organized_pages:
                organized_pages[page_type] = []
            organized_pages[page_type].append(page)
        
        # Step 4: Display organization
        for page_type, page_list in organized_pages.items():
            print(f"{page_type}: {len(page_list)} pages")
            
    except Exception as e:
        print(f"Error loading Notion pages: {e}")
        pages = []
    
    return pages

def advanced_youtube_transcript_loader():
    """
    Load YouTube video transcripts with metadata
    Use this for: Educational content, meeting recordings
    """
    # Step 1: Set up YouTube loader
    video_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=example123"
    ]
    
    documents = []
    
    for url in video_urls:
        try:
            # Step 2: Load transcript for each video
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,        # Include video metadata
                language=["en", "auto"],    # Preferred languages
                translation="en"            # Translate to English if needed
            )
            
            # Step 3: Load and process
            video_docs = loader.load()
            
            for doc in video_docs:
                # Step 4: Add custom metadata
                doc.metadata.update({
                    "content_type": "youtube_transcript",
                    "processed_date": "2024-01-15",
                    "language": "en"
                })
                
                # Step 5: Clean transcript content
                content = doc.page_content
                # Remove timestamps and clean formatting
                import re
                content = re.sub(r'\[\d+:\d+\]', '', content)
                content = ' '.join(content.split())
                doc.page_content = content
            
            documents.extend(video_docs)
            print(f"Loaded transcript from {url}")
            
        except Exception as e:
            print(f"Error loading {url}: {e}")
    
    return documents

def advanced_custom_loader_with_chunking():
    """
    Custom loader with intelligent text chunking
    Use this for: Large documents, optimized retrieval
    """
    # Step 1: Load documents from multiple sources
    all_documents = []
    
    # Load from different sources
    loaders = [
        TextLoader("large_document.txt", encoding="utf-8"),
        PyPDFLoader("research_paper.pdf"),
        # Add more loaders as needed
    ]
    
    for loader in loaders:
        try:
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error with loader {loader}: {e}")
    
    # Step 2: Set up intelligent text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,              # Target chunk size
        chunk_overlap=200,            # Overlap between chunks
        length_function=len,          # Function to measure length
        separators=[               # Hierarchy of separators
            "\n\n",                   # Paragraph breaks
            "\n",                     # Line breaks
            " ",                      # Spaces
            ""                        # Character level (last resort)
        ],
        keep_separator=True,          # Keep separators in chunks
        add_start_index=True          # Add start position metadata
    )
    
    # Step 3: Split documents into optimized chunks
    chunked_documents = text_splitter.split_documents(all_documents)
    
    # Step 4: Add enhanced metadata
    for i, chunk in enumerate(chunked_documents):
        chunk.metadata.update({
            "chunk_id": f"chunk_{i:06d}",
            "chunk_size": len(chunk.page_content),
            "processing_date": "2024-01-15",
            "chunking_strategy": "recursive_character"
        })
    
    # Step 5: Quality checks
    print(f"Original documents: {len(all_documents)}")
    print(f"Generated chunks: {len(chunked_documents)}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunked_documents) / len(chunked_documents):.0f}")
    
    return chunked_documents

def advanced_multi_format_processor():
    """
    Process multiple file formats with unified handling
    Use this for: Mixed content processing, document management systems
    """
    # Step 1: Define format-specific loaders
    format_loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.html': UnstructuredHTMLLoader,
        '.csv': CSVLoader
    }
    
    # Step 2: Process directory with format detection
    directory_path = Path("./mixed_documents")
    processed_documents = []
    
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext in format_loaders:
                # Step 3: Use appropriate loader
                loader_class = format_loaders[file_ext]
                
                try:
                    # Step 4: Configure loader based on format
                    if file_ext == '.csv':
                        loader = loader_class(str(file_path), encoding="utf-8")
                    else:
                        loader = loader_class(str(file_path))
                    
                    # Step 5: Load and enhance metadata
                    documents = loader.load()
                    
                    for doc in documents:
                        doc.metadata.update({
                            "file_type": file_ext,
                            "file_name": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "processed_timestamp": "2024-01-15T10:30:00Z"
                        })
                    
                    processed_documents.extend(documents)
                    print(f"Processed {file_path.name}: {len(documents)} documents")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Unsupported format: {file_ext}")
    
    # Step 6: Final processing and organization
    print(f"\nTotal processed documents: {len(processed_documents)}")
    
    # Group by file type
    by_type = {}
    for doc in processed_documents:
        file_type = doc.metadata.get("file_type", "unknown")
        by_type[file_type] = by_type.get(file_type, 0) + 1
    
    print("Documents by type:")
    for file_type, count in by_type.items():
        print(f"  {file_type}: {count}")
    
    return processed_documents

# =============================================================================
# UTILITY FUNCTIONS - Helper methods for common tasks
# =============================================================================

def create_sample_documents():
    """
    Create sample documents for testing
    Use this for: Testing, development, examples
    """
    # Create sample text file
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample document for LangChain testing.\n")
        f.write("It contains multiple lines and paragraphs.\n")
        f.write("Perfect for demonstrating document loading capabilities.")
    
    # Create sample CSV
    import csv
    with open("data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "description"])
        writer.writerow([1, "Product A", "High-quality product with great features"])
        writer.writerow([2, "Product B", "Affordable option for budget-conscious users"])
    
    print("Sample files created: example.txt, data.csv")

def validate_documents(documents):
    """
    Validate loaded documents for quality and completeness
    Use this for: Quality assurance, debugging
    """
    print(f"Validating {len(documents)} documents...")
    
    issues = []
    
    for i, doc in enumerate(documents):
        # Check for empty content
        if not doc.page_content.strip():
            issues.append(f"Document {i}: Empty content")
        
        # Check for missing metadata
        if not doc.metadata:
            issues.append(f"Document {i}: No metadata")
        
        # Check content length
        if len(doc.page_content) > 10000:
            issues.append(f"Document {i}: Very long content ({len(doc.page_content)} chars)")
    
    if issues:
        print("Issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
    else:
        print("All documents validated successfully!")
    
    return len(issues) == 0

# =============================================================================
# EXAMPLE USAGE - How to use these functions
# =============================================================================

if __name__ == "__main__":
    # Create sample files for testing
    create_sample_documents()
    
    print("=== Basic Examples ===")
    # Run basic examples
    text_docs = basic_text_loader()
    csv_docs = basic_csv_loader()
    
    print("\n=== Intermediate Examples ===")
    # Run intermediate examples
    dir_docs = intermediate_directory_loader()
    
    print("\n=== Advanced Examples ===")
    # Run advanced examples
    chunked_docs = advanced_custom_loader_with_chunking()
    
    # Validate all loaded documents
    all_docs = text_docs + csv_docs + chunked_docs
    validate_documents(all_docs)
    
    print(f"\nTotal documents processed: {len(all_docs)}")
