# LangChain Text Splitters: From Basics to Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Text Splitters](#basic-text-splitters)
3. [Character-Based Splitters](#character-based-splitters)
4. [Token-Based Splitters](#token-based-splitters)
5. [Document-Specific Splitters](#document-specific-splitters)
6. [Advanced Semantic Splitters](#advanced-semantic-splitters)
7. [Custom Text Splitters](#custom-text-splitters)
8. [Best Practices](#best-practices)

## Introduction

Text splitters in LangChain are essential components for breaking down large documents into smaller, manageable chunks for processing by language models. They help maintain context while ensuring chunks fit within token limits.

## Basic Text Splitters

### 1. CharacterTextSplitter

The most basic splitter that splits text based on a specified separator.

```python
from langchain.text_splitter import CharacterTextSplitter

# Basic usage
text = """
This is a long document that needs to be split into smaller chunks.
Each chunk should be manageable for processing by language models.
We want to maintain context while keeping chunks within reasonable size limits.
"""

# Initialize the splitter
splitter = CharacterTextSplitter(
    separator="\n",          # Split on newlines
    chunk_size=100,          # Maximum characters per chunk
    chunk_overlap=20,        # Overlap between chunks to maintain context
    length_function=len,     # Function to measure chunk length
)

# Split the text
chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

**Explanation:**
- `separator`: The character(s) to split on (newline in this case)
- `chunk_size`: Maximum size of each chunk in characters
- `chunk_overlap`: Number of characters to overlap between chunks for context preservation
- `length_function`: Function used to measure chunk size (default is `len()`)

### 2. RecursiveCharacterTextSplitter

A more intelligent splitter that tries multiple separators in order of preference.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample document with different structures
text = """
# Introduction
This is the introduction section of our document.

## Subsection 1
Here we discuss the first topic in detail.
It spans multiple paragraphs and includes:
- Important point 1
- Important point 2
- Important point 3

## Subsection 2
This section covers another important topic.
The content here is also quite detailed and requires careful handling.
"""

# Initialize recursive splitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # Try these separators in order
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

chunks = splitter.split_text(text)
print(f"Generated {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)
    print(f"Length: {len(chunk)} characters")
```

**Explanation:**
The recursive splitter tries separators in order:
1. First tries to split on double newlines (paragraph breaks)
2. Then single newlines (line breaks)
3. Then spaces (word boundaries)
4. Finally individual characters if needed

This maintains document structure better than simple character splitting.

## Character-Based Splitters

### 3. Advanced CharacterTextSplitter with Custom Logic

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def custom_length_function(text):
    """Custom function to measure text length, e.g., by word count"""
    return len(text.split())

# Create splitter with custom length function
splitter = CharacterTextSplitter(
    separator=".",  # Split on sentences
    chunk_size=50,  # 50 words per chunk
    chunk_overlap=10,  # 10 words overlap
    length_function=custom_length_function,
    is_separator_regex=False,
)

# Working with documents
documents = [
    Document(page_content="First document content here. It has multiple sentences. Each sentence contains important information.", 
             metadata={"source": "doc1.txt"}),
    Document(page_content="Second document with different content. This also spans multiple sentences. We want to preserve context.", 
             metadata={"source": "doc2.txt"})
]

# Split documents
split_docs = splitter.split_documents(documents)
for i, doc in enumerate(split_docs):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Word count: {custom_length_function(doc.page_content)}")
```

**Explanation:**
- `custom_length_function`: Measures chunks by word count instead of characters
- `split_documents()`: Works with Document objects, preserving metadata
- `is_separator_regex`: Whether the separator should be treated as a regex pattern

## Token-Based Splitters

### 4. TokenTextSplitter

Splits text based on token count, which is crucial for language model compatibility.

```python
from langchain.text_splitter import TokenTextSplitter

# Initialize token-based splitter
splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4 tokenizer
    chunk_size=100,               # 100 tokens per chunk
    chunk_overlap=20,             # 20 tokens overlap
)

text = """
Large language models have revolutionized natural language processing.
They can understand context, generate human-like text, and perform various tasks.
However, they have token limits that require careful text chunking strategies.
Token-based splitting ensures chunks fit within model constraints while preserving meaning.
"""

chunks = splitter.split_text(text)
print(f"Created {len(chunks)} token-based chunks:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"Content: {chunk}")
    print(f"Estimated tokens: {len(chunk.split()) * 1.3:.0f}")  # Rough estimate
```

### 5. Tiktoken Splitter (OpenAI-specific)

```python
from langchain.text_splitter import CharacterTextSplitter
import tiktoken

def tiktoken_len(text):
    """Calculate actual token count using tiktoken"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# Create splitter using actual token counting
splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,  # 50 tokens
    chunk_overlap=10,
    length_function=tiktoken_len,
)

text = "Your long text here that needs precise token-based splitting for OpenAI models..."
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    actual_tokens = tiktoken_len(chunk)
    print(f"Chunk {i+1}: {actual_tokens} tokens")
    print(f"Content: {chunk[:100]}...")
```

**Explanation:**
- Uses OpenAI's tiktoken library for precise token counting
- Essential for ensuring chunks fit within specific model token limits
- `cl100k_base` is the encoding used by GPT-4 and GPT-3.5-turbo

## Document-Specific Splitters

### 6. Markdown Splitter

Specialized for Markdown documents, preserving structure.

```python
from langchain.text_splitter import MarkdownTextSplitter

markdown_text = """
# Main Title

This is the introduction paragraph.

## Section 1: Overview

Here we provide an overview of the topic.

### Subsection 1.1

Detailed information about subsection 1.1.

```python
# Code example
def example_function():
    return "Hello, World!"
```

## Section 2: Implementation

This section covers implementation details.

- Point 1
- Point 2  
- Point 3
"""

splitter = MarkdownTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
)

chunks = splitter.split_text(markdown_text)
print(f"Markdown chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Markdown Chunk {i+1} ---")
    print(chunk)
```

### 7. Python Code Splitter

Designed specifically for Python code, maintaining syntactic integrity.

```python
from langchain.text_splitter import PythonCodeTextSplitter

python_code = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    """A class for processing data"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        """Load data from file"""
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def preprocess(self):
        """Preprocess the data"""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Remove null values
        self.data = self.data.dropna()
        
        # Normalize numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_cols] = (self.data[numerical_cols] - self.data[numerical_cols].mean()) / self.data[numerical_cols].std()
        
        return self.data

def main():
    processor = DataProcessor("data.csv")
    data = processor.load_data()
    processed_data = processor.preprocess()
    print("Data processing completed")

if __name__ == "__main__":
    main()
'''

splitter = PythonCodeTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_text(python_code)
print(f"Python code chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Code Chunk {i+1} ---")
    print(chunk)
    print("=" * 50)
```

**Explanation:**
- Preserves Python syntax and structure
- Tries to split at natural boundaries (functions, classes)
- Maintains code readability and executability

## Advanced Semantic Splitters

### 8. Semantic Chunking with Embeddings

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticTextSplitter:
    def __init__(self, embeddings_model, similarity_threshold=0.8):
        self.embeddings = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
        )
    
    def split_text_semantically(self, text):
        # First, split into base chunks
        base_chunks = self.base_splitter.split_text(text)
        
        if len(base_chunks) <= 1:
            return base_chunks
        
        # Get embeddings for each chunk
        embeddings = self.embeddings.embed_documents(base_chunks)
        
        # Group similar chunks together
        semantic_chunks = []
        current_group = [base_chunks[0]]
        
        for i in range(1, len(base_chunks)):
            # Calculate similarity with previous chunk
            similarity = cosine_similarity(
                [embeddings[i-1]], 
                [embeddings[i]]
            )[0][0]
            
            if similarity >= self.similarity_threshold:
                current_group.append(base_chunks[i])
            else:
                # Start new group
                semantic_chunks.append(" ".join(current_group))
                current_group = [base_chunks[i]]
        
        # Add the last group
        semantic_chunks.append(" ".join(current_group))
        
        return semantic_chunks

# Example usage (requires OpenAI API key)
# embeddings = OpenAIEmbeddings()
# semantic_splitter = SemanticTextSplitter(embeddings)

# Sample text with different topics
mixed_text = """
Machine learning is a subset of artificial intelligence that focuses on algorithms.
Deep learning uses neural networks with multiple layers.
Neural networks are inspired by biological neurons in the brain.

The weather today is sunny and warm.
It's a perfect day for outdoor activities.
Many people enjoy hiking when the weather is nice.

Python is a popular programming language.
It's widely used for data science and web development.
Python's syntax is clean and readable.
"""

# This would work with actual embeddings
# chunks = semantic_splitter.split_text_semantically(mixed_text)
```

### 9. Sliding Window Splitter

Creates overlapping windows for better context preservation.

```python
class SlidingWindowSplitter:
    def __init__(self, window_size=3, step_size=1):
        self.window_size = window_size
        self.step_size = step_size
    
    def split_sentences(self, text):
        """Split text into sentences and create sliding windows"""
        import re
        
        # Simple sentence splitting (could use more sophisticated methods)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        windows = []
        for i in range(0, len(sentences) - self.window_size + 1, self.step_size):
            window = sentences[i:i + self.window_size]
            windows.append(". ".join(window) + ".")
        
        return windows

# Example usage
text = """
The sun rises in the east. It provides light and warmth to our planet. 
Plants use sunlight for photosynthesis. This process converts carbon dioxide into oxygen. 
Animals depend on plants for survival. The ecosystem is interconnected. 
Human activities affect natural balance. We must protect our environment.
"""

splitter = SlidingWindowSplitter(window_size=3, step_size=2)
windows = splitter.split_sentences(text)

print(f"Created {len(windows)} sliding windows:")
for i, window in enumerate(windows):
    print(f"\nWindow {i+1}:")
    print(window)
```

## Custom Text Splitters

### 10. Domain-Specific Custom Splitter

```python
from langchain.text_splitter import TextSplitter
from typing import List
import re

class LegalDocumentSplitter(TextSplitter):
    """Custom splitter for legal documents"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Legal document patterns
        self.section_pattern = r'\n\s*(?:Section|Article|Chapter)\s+\d+'
        self.subsection_pattern = r'\n\s*\(\w+\)'
        self.paragraph_pattern = r'\n\s*\d+\.'
    
    def split_text(self, text: str) -> List[str]:
        """Split legal text preserving structure"""
        
        # First try to split by sections
        sections = re.split(self.section_pattern, text)
        chunks = []
        
        for section in sections:
            if len(section.strip()) == 0:
                continue
                
            if len(section) <= self.chunk_size:
                chunks.append(section.strip())
            else:
                # Split large sections by subsections
                subsections = re.split(self.subsection_pattern, section)
                for subsection in subsections:
                    if len(subsection.strip()) == 0:
                        continue
                        
                    if len(subsection) <= self.chunk_size:
                        chunks.append(subsection.strip())
                    else:
                        # Further split by paragraphs
                        paragraphs = re.split(self.paragraph_pattern, subsection)
                        for para in paragraphs:
                            if para.strip():
                                chunks.append(para.strip())
        
        return chunks

# Example usage
legal_text = """
Section 1. Definitions

For purposes of this agreement, the following terms shall have the meanings set forth below:

(a) "Company" means the corporation entering into this agreement.
(b) "Employee" means any individual employed by the Company.
(c) "Confidential Information" means any proprietary information.

Section 2. Obligations

1. Employee agrees to maintain confidentiality.
2. Employee shall not disclose information to third parties.
3. Employee must return all company property upon termination.
"""

legal_splitter = LegalDocumentSplitter(chunk_size=200)
chunks = legal_splitter.split_text(legal_text)

print(f"Legal document chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Legal Chunk {i+1} ---")
    print(chunk)
```

### 11. Conversational Splitter

Designed for chat logs and conversational data.

```python
class ConversationSplitter:
    def __init__(self, max_exchanges=5, overlap_exchanges=1):
        self.max_exchanges = max_exchanges
        self.overlap_exchanges = overlap_exchanges
    
    def split_conversation(self, conversation_text):
        """Split conversation maintaining speaker context"""
        
        # Parse conversation into exchanges
        lines = conversation_text.strip().split('\n')
        exchanges = []
        current_speaker = None
        current_message = []
        
        for line in lines:
            if ':' in line and len(line.split(':', 1)) == 2:
                # New speaker
                if current_speaker and current_message:
                    exchanges.append({
                        'speaker': current_speaker,
                        'message': ' '.join(current_message).strip()
                    })
                
                speaker, message = line.split(':', 1)
                current_speaker = speaker.strip()
                current_message = [message.strip()]
            else:
                # Continuation of current message
                if line.strip():
                    current_message.append(line.strip())
        
        # Add the last exchange
        if current_speaker and current_message:
            exchanges.append({
                'speaker': current_speaker,
                'message': ' '.join(current_message).strip()
            })
        
        # Group exchanges into chunks
        chunks = []
        for i in range(0, len(exchanges), self.max_exchanges - self.overlap_exchanges):
            chunk_exchanges = exchanges[i:i + self.max_exchanges]
            chunk_text = '\n'.join([f"{ex['speaker']}: {ex['message']}" for ex in chunk_exchanges])
            chunks.append(chunk_text)
        
        return chunks

# Example conversation
conversation = """
Alice: Hi there! How are you doing today?
Bob: I'm doing great, thanks for asking. Just working on a new project.
Alice: That sounds interesting! What kind of project is it?
Bob: It's a machine learning model for text analysis.
I've been working on it for the past few weeks.
Alice: Machine learning is fascinating. Are you using any specific frameworks?
Bob: Yes, I'm primarily using scikit-learn and transformers.
The results have been quite promising so far.
Alice: That's awesome! I'd love to hear more about your approach.
"""

conv_splitter = ConversationSplitter(max_exchanges=3, overlap_exchanges=1)
chunks = conv_splitter.split_conversation(conversation)

print(f"Conversation chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Conversation Chunk {i+1} ---")
    print(chunk)
```

## Best Practices

### 12. Intelligent Chunk Size Selection

```python
def optimize_chunk_size(text, target_model="gpt-3.5-turbo"):
    """Determine optimal chunk size based on content and model"""
    
    model_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "claude": 100000,
    }
    
    max_tokens = model_limits.get(target_model, 4096)
    
    # Reserve tokens for prompt and response
    available_tokens = max_tokens - 1000
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_chunk_size = available_tokens * 4
    
    # Adjust based on text characteristics
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
    if avg_word_length > 6:  # Technical text
        estimated_chunk_size = int(estimated_chunk_size * 0.8)
    
    return min(estimated_chunk_size, 2000)  # Cap at reasonable size

def create_adaptive_splitter(text, target_model="gpt-3.5-turbo"):
    """Create splitter adapted to content and model"""
    
    chunk_size = optimize_chunk_size(text, target_model)
    overlap = min(chunk_size // 10, 200)  # 10% overlap, max 200 chars
    
    # Choose splitter based on content type
    if "```" in text or "def " in text or "class " in text:
        # Code content
        splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    elif "#" in text and text.count("#") > 3:
        # Markdown content
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    else:
        # General text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    return splitter

# Example usage
sample_text = "Your document content here..."
adaptive_splitter = create_adaptive_splitter(sample_text, "gpt-4")
chunks = adaptive_splitter.split_text(sample_text)
```

### 13. Quality Assessment and Metrics

```python
def assess_chunk_quality(chunks):
    """Assess the quality of text chunks"""
    
    metrics = {
        'total_chunks': len(chunks),
        'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks),
        'size_variance': 0,
        'empty_chunks': sum(1 for chunk in chunks if not chunk.strip()),
        'very_short_chunks': sum(1 for chunk in chunks if len(chunk) < 50),
        'very_long_chunks': sum(1 for chunk in chunks if len(chunk) > 2000),
    }
    
    # Calculate size variance
    avg_size = metrics['avg_chunk_size']
    variance = sum((len(chunk) - avg_size) ** 2 for chunk in chunks) / len(chunks)
    metrics['size_variance'] = variance ** 0.5
    
    return metrics

def print_chunk_analysis(chunks):
    """Print detailed analysis of chunks"""
    metrics = assess_chunk_quality(chunks)
    
    print("=== Chunk Quality Analysis ===")
    print(f"Total chunks: {metrics['total_chunks']}")
    print(f"Average chunk size: {metrics['avg_chunk_size']:.1f} characters")
    print(f"Size standard deviation: {metrics['size_variance']:.1f}")
    print(f"Empty chunks: {metrics['empty_chunks']}")
    print(f"Very short chunks (<50 chars): {metrics['very_short_chunks']}")
    print(f"Very long chunks (>2000 chars): {metrics['very_long_chunks']}")
    
    # Quality score (simple heuristic)
    quality_score = 100
    quality_score -= metrics['empty_chunks'] * 10
    quality_score -= metrics['very_short_chunks'] * 5
    quality_score -= metrics['very_long_chunks'] * 5
    quality_score -= min(metrics['size_variance'] / 100, 20)
    
    print(f"Quality score: {max(0, quality_score):.1f}/100")

# Example usage
text = "Your long document here..."
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print_chunk_analysis(chunks)
```

## Conclusion

This guide covers the spectrum of text splitting techniques in LangChain, from basic character-based splitting to advanced semantic approaches. The key is choosing the right splitter for your specific use case:

- **RecursiveCharacterTextSplitter**: Best general-purpose option
- **TokenTextSplitter**: When precise token control is needed
- **Document-specific splitters**: For structured content (Markdown, code, etc.)
- **Custom splitters**: For domain-specific requirements
- **Semantic splitters**: When meaning preservation is critical

Remember to always test your splitting strategy with representative data and adjust parameters based on your specific requirements and target models.