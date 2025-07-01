# LangChain Text Splitters: From Basic to Advanced
# Complete guide with practical examples and detailed comments

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    PythonCodeTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    LatexTextSplitter,
    Language
)
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
import tiktoken
import re
from typing import List, Dict, Any

# =============================================================================
# BASIC EXAMPLES - Getting Started with Text Splitting
# =============================================================================

def basic_character_splitter():
    """
    Basic character-based text splitting
    Use this for: Simple text chunking, uniform chunk sizes
    """
    # Sample text for demonstration
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and reason about their environment.
    The framework consists of several parts: LangChain Libraries, LangChain Templates,
    LangServe, and LangSmith. Each component serves a specific purpose in the development
    of language model applications. LangChain provides tools for prompt management,
    chains, data augmented generation, agents, memory, and evaluation.
    """
    
    # Step 1: Create a basic character text splitter
    splitter = CharacterTextSplitter(
        chunk_size=100,        # Maximum characters per chunk
        chunk_overlap=20,      # Characters to overlap between chunks
        length_function=len,   # Function to measure chunk length
        separator="\n"         # Primary separator to split on
    )
    
    # Step 2: Split the text
    chunks = splitter.split_text(sample_text)
    
    # Step 3: Display results
    print("=== Basic Character Splitter ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"'{chunk.strip()}'")
    
    return chunks

def basic_recursive_splitter():
    """
    Recursive character splitter - tries multiple separators
    Use this for: General text processing, better chunk boundaries
    """
    sample_text = """
    # Introduction to LangChain
    
    LangChain is a powerful framework for building applications with language models.
    
    ## Key Components
    
    1. **LangChain Libraries**: Core building blocks
    2. **LangChain Templates**: Reference architectures  
    3. **LangServe**: Deployment platform
    4. **LangSmith**: Developer platform
    
    ## Getting Started
    
    To begin using LangChain, you'll need to install the package and understand
    the basic concepts. The framework provides abstractions for working with
    language models in a structured way.
    """
    
    # Step 1: Create recursive character text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,              # Target chunk size
        chunk_overlap=50,            # Overlap between chunks
        length_function=len,         # How to measure chunk length
        separators=[                 # List of separators to try (in order)
            "\n\n",                  # Double newlines (paragraphs)
            "\n",                    # Single newlines
            " ",                     # Spaces
            ""                       # Characters (last resort)
        ]
    )
    
    # Step 2: Split the text
    chunks = splitter.split_text(sample_text)
    
    # Step 3: Analyze results
    print("\n=== Basic Recursive Splitter ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"'{chunk.strip()}'")
        # Show what separator was likely used
        if '\n\n' in chunk:
            print("  → Split on paragraph breaks")
        elif '\n' in chunk:
            print("  → Split on line breaks")
        else:
            print("  → Split on spaces/characters")
    
    return chunks

def basic_document_splitting():
    """
    Splitting LangChain Document objects (with metadata)
    Use this for: Processing loaded documents while preserving metadata
    """
    # Step 1: Create sample documents
    documents = [
        Document(
            page_content="This is the first document about AI and machine learning. "
                        "It contains important information about neural networks and deep learning. "
                        "The content spans multiple sentences and covers various topics.",
            metadata={"source": "ai_guide.txt", "chapter": 1}
        ),
        Document(
            page_content="The second document focuses on natural language processing. "
                        "It explains tokenization, embeddings, and transformer architectures. "
                        "This document is essential for understanding modern NLP techniques.",
            metadata={"source": "nlp_guide.txt", "chapter": 2}
        )
    ]
    
    # Step 2: Create splitter for documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        add_start_index=True  # Add start position to metadata
    )
    
    # Step 3: Split documents (preserves metadata)
    split_documents = splitter.split_documents(documents)
    
    # Step 4: Examine results
    print("\n=== Basic Document Splitting ===")
    print(f"Original documents: {len(documents)}")
    print(f"Split documents: {len(split_documents)}")
    
    for i, doc in enumerate(split_documents):
        print(f"\nSplit Document {i+1}:")
        print(f"Content ({len(doc.page_content)} chars): '{doc.page_content[:100]}...'")
        print(f"Metadata: {doc.metadata}")
    
    return split_documents

# =============================================================================
# INTERMEDIATE EXAMPLES - Token-Based and Language-Specific Splitting
# =============================================================================

def intermediate_token_splitter():
    """
    Token-based splitting using tiktoken (OpenAI tokenizer)
    Use this for: API cost optimization, model context limits
    """
    sample_text = """
    Artificial Intelligence (AI) has revolutionized numerous industries and continues
    to shape the future of technology. Machine learning, a subset of AI, enables
    computers to learn and improve from experience without being explicitly programmed.
    Deep learning, which uses neural networks with multiple layers, has achieved
    remarkable success in image recognition, natural language processing, and game playing.
    The applications of AI are vast, ranging from autonomous vehicles and medical
    diagnosis to recommendation systems and virtual assistants.
    """
    
    # Step 1: Create token-based splitter
    splitter = TokenTextSplitter(
        chunk_size=50,          # Maximum tokens per chunk
        chunk_overlap=10,       # Tokens to overlap
        encoding_name="cl100k_base",  # OpenAI's encoding (GPT-3.5/4)
        model_name="gpt-3.5-turbo",   # Specific model for accurate counting
        allowed_special=set(),   # Special tokens to allow
        disallowed_special="all" # Raise error on special tokens
    )
    
    # Step 2: Split the text
    chunks = splitter.split_text(sample_text)
    
    # Step 3: Analyze token usage
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    print("\n=== Intermediate Token Splitter ===")
    print(f"Original text: {len(encoding.encode(sample_text))} tokens")
    print(f"Number of chunks: {len(chunks)}")
    
    total_tokens = 0
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        total_tokens += token_count
        print(f"\nChunk {i+1}: {token_count} tokens")
        print(f"Content: '{chunk.strip()[:100]}...'")
    
    print(f"\nTotal tokens after splitting: {total_tokens}")
    print(f"Overhead from overlapping: {total_tokens - len(encoding.encode(sample_text))} tokens")
    
    return chunks

def intermediate_spacy_splitter():
    """
    Sentence-based splitting using spaCy NLP
    Use this for: Linguistically aware splitting, better semantic boundaries
    """
    # Note: Requires 'pip install spacy' and 'python -m spacy download en_core_web_sm'
    
    sample_text = """
    Natural Language Processing (NLP) is a field of AI. It focuses on interaction between
    computers and humans. NLP combines computational linguistics with machine learning.
    Common NLP tasks include tokenization, part-of-speech tagging, and named entity recognition.
    Advanced applications involve sentiment analysis, machine translation, and question answering.
    Modern NLP relies heavily on transformer architectures like BERT and GPT.
    """
    
    try:
        # Step 1: Create spaCy-based splitter
        splitter = SpacyTextSplitter(
            chunk_size=100,         # Target character count
            chunk_overlap=20,       # Character overlap
            pipeline="en_core_web_sm",  # spaCy model
            separator=" "           # Fallback separator
        )
        
        # Step 2: Split text using sentence boundaries
        chunks = splitter.split_text(sample_text)
        
        # Step 3: Show sentence-aware splitting
        print("\n=== Intermediate spaCy Splitter ===")
        print(f"Original text length: {len(sample_text)} characters")
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            # Count sentences in chunk
            sentence_count = chunk.count('.') + chunk.count('!') + chunk.count('?')
            print(f"\nChunk {i+1} ({len(chunk)} chars, ~{sentence_count} sentences):")
            print(f"'{chunk.strip()}'")
            
    except ImportError:
        print("\n=== spaCy Splitter ===")
        print("spaCy not installed. Install with: pip install spacy")
        print("Then download model: python -m spacy download en_core_web_sm")
        chunks = []
    
    return chunks

def intermediate_nltk_splitter():
    """
    NLTK-based sentence splitting
    Use this for: Academic text processing, research applications
    """
    # Note: Requires 'pip install nltk' and downloading punkt tokenizer
    
    sample_text = """
    The field of machine learning has evolved significantly over the past decade.
    Supervised learning algorithms require labeled training data. Unsupervised learning
    discovers patterns in unlabeled data. Reinforcement learning learns through
    interaction with an environment. Deep learning has achieved state-of-the-art
    results in many domains. Transfer learning allows models to leverage pre-trained
    knowledge for new tasks.
    """
    
    try:
        # Step 1: Create NLTK-based splitter
        splitter = NLTKTextSplitter(
            chunk_size=120,         # Target character count
            chunk_overlap=25,       # Character overlap
            separator=" "           # Fallback separator
        )
        
        # Step 2: Split using NLTK sentence tokenizer
        chunks = splitter.split_text(sample_text)
        
        # Step 3: Analyze sentence boundaries
        print("\n=== Intermediate NLTK Splitter ===")
        print(f"Original text length: {len(sample_text)} characters")
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            # Simple sentence counting
            sentence_count = len([s for s in chunk.split('.') if s.strip()])
            print(f"\nChunk {i+1} ({len(chunk)} chars, ~{sentence_count} sentences):")
            print(f"'{chunk.strip()}'")
            
    except ImportError:
        print("\n=== NLTK Splitter ===")
        print("NLTK not installed. Install with: pip install nltk")
        print("Then run: import nltk; nltk.download('punkt')")
        chunks = []
    
    return chunks

# =============================================================================
# ADVANCED EXAMPLES - Specialized and Custom Splitters
# =============================================================================

def advanced_code_splitter():
    """
    Python code-aware text splitting
    Use this for: Processing code documentation, API references
    """
    # Sample Python code
    sample_code = '''
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize the input texts
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    # Create sets for comparison
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

class TextProcessor:
    """Process and analyze text data."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.processed_count = 0
    
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """Process a batch of texts."""
        results = []
        for text in texts:
            result = self.process_single(text)
            results.append(result)
        return results
    '''
    
    # Step 1: Create Python code splitter
    splitter = PythonCodeTextSplitter(
        chunk_size=300,         # Target character count
        chunk_overlap=50        # Character overlap
    )
    
    # Step 2: Split code while respecting structure
    chunks = splitter.split_text(sample_code)
    
    # Step 3: Analyze code structure preservation
    print("\n=== Advanced Code Splitter ===")
    print(f"Original code length: {len(sample_code)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        
        # Analyze what's in this chunk
        if 'def ' in chunk:
            func_count = chunk.count('def ')
            print(f"  → Contains {func_count} function definition(s)")
        if 'class ' in chunk:
            class_count = chunk.count('class ')
            print(f"  → Contains {class_count} class definition(s)")
        if '"""' in chunk:
            print("  → Contains docstring")
        if '#' in chunk:
            comment_count = len([line for line in chunk.split('\n') if line.strip().startswith('#')])
            print(f"  → Contains {comment_count} comment line(s)")
        
        print(f"Preview: {chunk.strip()[:100]}...")
    
    return chunks

def advanced_markdown_splitter():
    """
    Markdown header-aware splitting
    Use this for: Documentation, structured content
    """
    # Sample Markdown content
    markdown_content = """
# LangChain Documentation

LangChain is a framework for developing applications powered by language models.

## Installation

To install LangChain, use pip:

```bash
pip install langchain
```

### Prerequisites

Make sure you have Python 3.8 or higher installed.

## Quick Start

Here's a simple example to get you started:

```python
from langchain.llms import OpenAI
llm = OpenAI()
result = llm("What is LangChain?")
print(result)
```

### Configuration

You'll need to set up your API keys:

1. Create a `.env` file
2. Add your OpenAI API key
3. Load environment variables

## Advanced Usage

### Chains

Chains allow you to combine multiple components:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = "What is a good name for a {product}?"
prompt = PromptTemplate(template=template, input_variables=["product"])
chain = LLMChain(llm=llm, prompt=prompt)
```

### Agents

Agents can use tools to interact with the world:

```python
from langchain.agents import load_tools, initialize_agent

tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```

## Conclusion

This guide covers the basics of LangChain. For more information, check the official documentation.
    """
    
    # Step 1: Create markdown header splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in content
    )
    
    # Step 2: Split by headers first
    md_header_splits = markdown_splitter.split_text(markdown_content)
    
    # Step 3: Further split long sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    
    # Step 4: Analyze hierarchical structure
    print("\n=== Advanced Markdown Splitter ===")
    print(f"Original markdown length: {len(markdown_content)} characters")
    print(f"Header-based splits: {len(md_header_splits)}")
    print(f"Final splits: {len(final_splits)}")
    
    for i, doc in enumerate(final_splits):
        print(f"\nSplit {i+1} ({len(doc.page_content)} chars):")
        print(f"Metadata: {doc.metadata}")
        
        # Show content structure
        content_preview = doc.page_content.strip()[:150]
        if content_preview.startswith('#'):
            header_level = len(content_preview.split()[0])
            print(f"  → Starts with Header Level {header_level}")
        if '```' in content_preview:
            print(f"  → Contains code block")
        
        print(f"Preview: {content_preview}...")
    
    return final_splits

def advanced_html_splitter():
    """
    HTML header-aware splitting
    Use this for: Web content, HTML documentation
    """
    # Sample HTML content
    html_content = """
    <html>
    <body>
    <h1>Web Development Guide</h1>
    <p>This guide covers modern web development practices.</p>
    
    <h2>Frontend Technologies</h2>
    <p>Frontend development involves creating user interfaces that users interact with directly.</p>
    
    <h3>HTML</h3>
    <p>HTML (HyperText Markup Language) is the standard markup language for creating web pages.
    It provides the basic structure and content of web pages using elements and tags.</p>
    
    <h3>CSS</h3>
    <p>CSS (Cascading Style Sheets) is used for styling HTML elements.
    It controls the presentation, layout, and visual appearance of web pages.</p>
    
    <h3>JavaScript</h3>
    <p>JavaScript is a programming language that enables interactive web pages.
    It's an essential part of web applications alongside HTML and CSS.</p>
    
    <h2>Backend Technologies</h2>
    <p>Backend development focuses on server-side logic and database management.</p>
    
    <h3>Server Languages</h3>
    <p>Popular server-side languages include Python, JavaScript (Node.js), Java, and PHP.
    Each has its own strengths and use cases in web development.</p>
    
    <h3>Databases</h3>
    <p>Databases store and manage application data. Common types include SQL databases
    like PostgreSQL and MySQL, and NoSQL databases like MongoDB.</p>
    
    </body>
    </html>
    """
    
    # Step 1: Create HTML header splitter
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    
    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    # Step 2: Split by HTML headers
    html_header_splits = html_splitter.split_text(html_content)
    
    # Step 3: Further processing if needed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    
    final_splits = text_splitter.split_documents(html_header_splits)
    
    # Step 4: Analyze HTML structure
    print("\n=== Advanced HTML Splitter ===")
    print(f"Original HTML length: {len(html_content)} characters")
    print(f"Header-based splits: {len(html_header_splits)}")
    print(f"Final splits: {len(final_splits)}")
    
    for i, doc in enumerate(final_splits):
        print(f"\nSplit {i+1} ({len(doc.page_content)} chars):")
        print(f"Metadata: {doc.metadata}")
        
        # Clean content for preview
        content = doc.page_content.strip()
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        print(f"Preview: {content[:150]}...")
    
    return final_splits

def advanced_latex_splitter():
    """
    LaTeX document splitting with awareness of structure
    Use this for: Academic papers, mathematical documents
    """
    # Sample LaTeX content
    latex_content = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \title{Introduction to Machine Learning}
    \author{Research Team}
    
    \begin{document}
    \maketitle
    
    \section{Introduction}
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from and make predictions or decisions based on data.
    
    \subsection{Supervised Learning}
    In supervised learning, algorithms learn from labeled training data to make
    predictions on new, unseen data. Common supervised learning tasks include:
    
    \begin{itemize}
    \item Classification: Predicting discrete categories
    \item Regression: Predicting continuous values
    \end{itemize}
    
    The mathematical foundation involves minimizing a loss function:
    \begin{equation}
    L(\theta) = \frac{1}{n} \sum_{i=1}^{n} l(y_i, f(x_i; \theta))
    \end{equation}
    
    \subsection{Unsupervised Learning}
    Unsupervised learning algorithms find patterns in data without labeled examples.
    This includes clustering, dimensionality reduction, and anomaly detection.
    
    \section{Deep Learning}
    Deep learning uses neural networks with multiple layers to learn complex patterns.
    
    \subsection{Neural Networks}
    A neural network can be represented as:
    \begin{equation}
    y = f(W_n \cdot f(W_{n-1} \cdot ... \cdot f(W_1 \cdot x + b_1) + ... + b_{n-1}) + b_n)
    \end{equation}
    
    where $W_i$ are weight matrices and $b_i$ are bias vectors.
    
    \end{document}
    """
    
    # Step 1: Create LaTeX splitter
    latex_splitter = LatexTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Step 2: Split LaTeX content
    chunks = latex_splitter.split_text(latex_content)
    
    # Step 3: Analyze LaTeX structure
    print("\n=== Advanced LaTeX Splitter ===")
    print(f"Original LaTeX length: {len(latex_content)} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        
        # Analyze LaTeX elements
        if '\\section{' in chunk:
            print("  → Contains section header")
        if '\\subsection{' in chunk:
            print("  → Contains subsection header")
        if '\\begin{equation}' in chunk:
            print("  → Contains equation environment")
        if '\\begin{itemize}' in chunk:
            print("  → Contains itemize environment")
        if '$' in chunk:
            print("  → Contains inline math")
        
        # Clean preview
        preview = chunk.strip()[:200]
        preview = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '[LaTeX cmd]', preview)
        print(f"Preview: {preview}...")
    
    return chunks

def advanced_custom_semantic_splitter():
    """
    Custom semantic-aware splitter based on sentence embeddings
    Use this for: Semantic coherence, topic-based splitting
    """
    # This is a conceptual example - would require sentence-transformers
    # pip install sentence-transformers
    
    sample_text = """
    Climate change is one of the most pressing issues of our time. Rising global temperatures
    are causing ice caps to melt and sea levels to rise. This environmental crisis affects
    weather patterns worldwide, leading to more frequent extreme weather events.
    
    Artificial intelligence offers promising solutions to combat climate change. Machine learning
    algorithms can optimize energy consumption in smart grids. AI can also improve weather
    prediction models and help develop more efficient renewable energy systems.
    
    The economic impact of climate change is substantial. Industries must adapt to new
    regulations and changing consumer preferences. Green technologies are creating new
    job opportunities while traditional industries face challenges in transitioning
    to sustainable practices.
    """
    
    # Step 1: Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', sample_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print("\n=== Advanced Custom Semantic Splitter ===")
    print(f"Original text: {len(sample_text)} characters")
    print(f"Sentences: {len(sentences)}")
    
    try:
        # This would require actual sentence-transformers installation
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embeddings = model.encode(sentences)
        
        # For demonstration, we'll use a simpler keyword-based approach
        topics = {
            'climate': ['climate', 'temperature', 'weather', 'environmental', 'ice', 'sea'],
            'ai': ['artificial', 'intelligence', 'machine', 'learning', 'algorithms', 'AI'],
            'economic': ['economic', 'industries', 'job', 'green', 'technologies', 'sustainable']
        }
        
        # Step 2: Group sentences by topic
        topic_groups = {topic: [] for topic in topics}
        unassigned = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            assigned = False
            
            for topic, keywords in topics.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    topic_groups[topic].append(sentence)
                    assigned = True
                    break
            
            if not assigned:
                unassigned.append(sentence)
        
        # Step 3: Create semantic chunks
        semantic_chunks = []
        
        for topic, topic_sentences in topic_groups.items():
            if topic_sentences:
                chunk_content = ' '.join(topic_sentences)
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        'topic': topic,
                        'sentence_count': len(topic_sentences),
                        'splitting_method': 'semantic_keyword'
                    }
                )
                semantic_chunks.append(chunk_doc)
        
        if unassigned:
            chunk_content = ' '.join(unassigned)
            chunk_doc = Document(
                page_content=chunk_content,
                metadata={
                    'topic': 'unassigned',
                    'sentence_count': len(unassigned),
                    'splitting_method': 'semantic_keyword'
                }
            )
            semantic_chunks.append(chunk_doc)
        
        # Step 4: Display semantic grouping
        print(f"Semantic chunks created: {len(semantic_chunks)}")
        
        for i, chunk in enumerate(semantic_chunks):
            topic = chunk.metadata.get('topic', 'unknown')
            sentence_count = chunk.metadata.get('sentence_count', 0)
            
            print(f"\nTopic Chunk {i+1}: '{topic}' ({sentence_count} sentences)")
            print(f"Content ({len(chunk.page_content)} chars):")
            print(f"'{chunk.page_content[:150]}...'")
    
    except ImportError:
        print("For full semantic splitting, install: pip install sentence-transformers")
        print("Using simplified keyword-based approach for demonstration.")
        semantic_chunks = []
    
    return semantic_chunks

# =============================================================================
# UTILITY FUNCTIONS - Helper methods and best practices
# =============================================================================

def compare_splitters(text: str):
    """
    Compare different splitters on the same text
    Use this for: Choosing the right splitter, performance analysis
    """
    print("\n=== Splitter Comparison ===")
    
    splitters = {
        "Character": CharacterTextSplitter(chunk_size=200, chunk_overlap=50),
        "Recursive": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50),
        "Token": TokenTextSplitter(chunk_size=50, chunk_overlap=10, encoding_name="cl100k_base")
    }
    
    results = {}
    
    for name, splitter in splitters.items():
        try:
            chunks = splitter.split_text(text)
            
            # Calculate metrics
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            results[name] = {
                'chunks': len(chunks),
                'avg_size': avg_size,
                'min_size': min(chunk_sizes) if chunk_sizes else 0,
                'max_size': max(chunk_sizes) if chunk_sizes else 0,
                'total_chars': sum(chunk_sizes)
            }
            
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Display comparison
    print(f"Original text: {len(text)} characters\n