"""
LangChain Text Splitters: From Basics to Advanced
Complete Python implementation with examples and explanations

Requirements:
pip install langchain tiktoken openai scikit-learn numpy pandas

Author: AI Assistant
Date: 2025
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# LangChain imports
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TextSplitter
)
from langchain.docstore.document import Document

# Optional imports (install if needed)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


class TextSplitterExamples:
    """Comprehensive examples of LangChain text splitters"""
    
    def __init__(self):
        self.sample_texts = self._load_sample_texts()
    
    def _load_sample_texts(self) -> Dict[str, str]:
        """Load sample texts for different scenarios"""
        return {
            'general': """
This is a comprehensive document about artificial intelligence and machine learning.
The field has evolved rapidly over the past decade with significant breakthroughs.

Machine learning algorithms can now process vast amounts of data efficiently.
Deep learning models have achieved human-level performance in many tasks.
Natural language processing has seen remarkable improvements with transformer models.

The applications are endless: from healthcare to autonomous vehicles.
However, we must also consider the ethical implications of AI development.
Responsible AI development requires careful consideration of bias and fairness.
""",
            
            'markdown': """
# Artificial Intelligence Guide

This comprehensive guide covers the fundamentals of AI and machine learning.

## Introduction

Artificial Intelligence (AI) represents one of the most significant technological advances of our time.

### What is AI?

AI refers to the simulation of human intelligence in machines that are programmed to think and learn.

### Types of AI

There are several categories of AI:

- **Narrow AI**: Designed for specific tasks
- **General AI**: Hypothetical AI with human-level intelligence
- **Super AI**: AI that exceeds human intelligence

## Machine Learning

Machine learning is a subset of AI that focuses on algorithms that improve through experience.

### Supervised Learning

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Unsupervised Learning

Discovers patterns in data without labeled examples.

## Deep Learning

Deep learning uses neural networks with multiple layers to model complex patterns.
""",
            
            'code': '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class MLPipeline:
    """Complete machine learning pipeline"""
    
    def __init__(self, model_type='random_forest'):
        """Initialize the ML pipeline
        
        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str):
        """Preprocess the data for training
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Name of target column
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Handle missing values
        data = data.dropna()
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the machine learning model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model trained successfully")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report
        }
        
        return results

def main():
    """Main function to demonstrate the ML pipeline"""
    pipeline = MLPipeline('random_forest')
    
    # Load and preprocess data
    data = pipeline.load_data('dataset.csv')
    if data is not None:
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data, 'target')
        
        # Train and evaluate model
        pipeline.train_model(X_train, y_train)
        results = pipeline.evaluate_model(X_test, y_test)
        
        print(f"Model Accuracy: {results['accuracy']:.4f}")
        print("Classification Report:")
        print(results['classification_report'])

if __name__ == "__main__":
    main()
''',
            
            'legal': """
SECTION 1. DEFINITIONS

For purposes of this Agreement, the following terms shall have the meanings set forth below:

(a) "Company" means XYZ Corporation, a Delaware corporation.

(b) "Employee" means any individual who is employed by the Company on a full-time or part-time basis.

(c) "Confidential Information" means any and all proprietary information, trade secrets, and other confidential information.

SECTION 2. CONFIDENTIALITY OBLIGATIONS

1. Employee acknowledges that during employment, Employee may have access to Confidential Information.

2. Employee agrees to maintain the confidentiality of all Confidential Information.

3. Employee shall not disclose Confidential Information to any third party without prior written consent.

SECTION 3. NON-COMPETE PROVISIONS

Employee agrees that during the term of employment and for a period of twelve (12) months thereafter, Employee shall not engage in any business that competes with the Company.
""",
            
            'conversation': """
Alice: Hi Bob! How's your machine learning project coming along?

Bob: Hey Alice! It's going really well, thanks for asking. 
I've been working on a text classification model using transformers.
The initial results are quite promising.

Alice: That sounds fascinating! What kind of text are you classifying?

Bob: I'm working with customer support tickets.
The goal is to automatically categorize them by urgency and topic.
This should help our support team prioritize their work more effectively.

Alice: What a practical application! Are you using any specific frameworks?

Bob: Yes, I'm primarily using Hugging Face transformers with PyTorch.
I started with a pre-trained BERT model and fine-tuned it on our data.
The accuracy is around 85% so far, which is pretty good for a first iteration.

Alice: That's impressive! Have you considered trying other models like RoBERTa or DeBERTa?

Bob: Great suggestion! I actually have RoBERTa on my list to try next.
I've heard it can perform better on certain types of text classification tasks.
I'm also planning to experiment with some data augmentation techniques.
"""
        }
    
    def demo_basic_splitters(self):
        """Demonstrate basic text splitters"""
        print("=" * 60)
        print("BASIC TEXT SPLITTERS")
        print("=" * 60)
        
        text = self.sample_texts['general']
        
        # 1. Character Text Splitter
        print("\n1. CHARACTER TEXT SPLITTER")
        print("-" * 30)
        
        char_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=150,
            chunk_overlap=30,
            length_function=len,
        )
        
        chunks = char_splitter.split_text(text)
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print(f"'{chunk.strip()}'")
        
        # 2. Recursive Character Text Splitter
        print("\n\n2. RECURSIVE CHARACTER TEXT SPLITTER")
        print("-" * 40)
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = recursive_splitter.split_text(text)
        print(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk)} chars):")
            print(f"'{chunk.strip()}'")
    
    def demo_character_based_splitters(self):
        """Demonstrate advanced character-based splitters"""
        print("\n" + "=" * 60)
        print("CHARACTER-BASED SPLITTERS")
        print("=" * 60)
        
        def custom_length_function(text):
            """Custom function to measure text length by word count"""
            return len(text.split())
        
        # Custom length function example
        print("\n1. CUSTOM LENGTH FUNCTION (Word-based)")
        print("-" * 45)
        
        word_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=25,  # 25 words per chunk
            chunk_overlap=5,  # 5 words overlap
            length_function=custom_length_function,
        )
        
        text = self.sample_texts['general']
        chunks = word_splitter.split_text(text)
        
        print(f"Number of word-based chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            word_count = custom_length_function(chunk)
            print(f"\nChunk {i+1} ({word_count} words):")
            print(f"'{chunk.strip()}'")
        
        # Working with documents and metadata
        print("\n\n2. WORKING WITH DOCUMENTS AND METADATA")
        print("-" * 45)
        
        documents = [
            Document(
                page_content=self.sample_texts['general'][:300],
                metadata={"source": "ai_guide.txt", "page": 1}
            ),
            Document(
                page_content=self.sample_texts['general'][300:],
                metadata={"source": "ai_guide.txt", "page": 2}
            )
        ]
        
        doc_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=150,
            chunk_overlap=20,
        )
        
        split_docs = doc_splitter.split_documents(documents)
        
        print(f"Number of document chunks: {len(split_docs)}")
        for i, doc in enumerate(split_docs):
            print(f"\nDocument Chunk {i+1}:")
            print(f"Content: '{doc.page_content[:100]}...'")
            print(f"Metadata: {doc.metadata}")
    
    def demo_token_based_splitters(self):
        """Demonstrate token-based splitters"""
        print("\n" + "=" * 60)
        print("TOKEN-BASED SPLITTERS")
        print("=" * 60)
        
        text = self.sample_texts['general']
        
        # 1. Basic Token Splitter
        print("\n1. BASIC TOKEN TEXT SPLITTER")
        print("-" * 35)
        
        token_splitter = TokenTextSplitter(
            encoding_name="cl100k_base",  # GPT-4 tokenizer
            chunk_size=50,
            chunk_overlap=10,
        )
        
        chunks = token_splitter.split_text(text)
        print(f"Number of token-based chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            # Rough token estimate (actual count would require tiktoken)
            estimated_tokens = len(chunk.split()) * 1.3
            print(f"\nChunk {i+1} (~{estimated_tokens:.0f} tokens):")
            print(f"'{chunk.strip()}'")
        
        # 2. Tiktoken Integration (if available)
        if TIKTOKEN_AVAILABLE:
            print("\n\n2. TIKTOKEN INTEGRATION")
            print("-" * 25)
            
            def tiktoken_len(text):
                tokenizer = tiktoken.get_encoding("cl100k_base")
                tokens = tokenizer.encode(text)
                return len(tokens)
            
            tiktoken_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=30,  # 30 tokens
                chunk_overlap=5,
                length_function=tiktoken_len,
            )
            
            chunks = tiktoken_splitter.split_text(text)
            print(f"Number of tiktoken chunks: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                actual_tokens = tiktoken_len(chunk)
                print(f"\nChunk {i+1} ({actual_tokens} tokens):")
                print(f"'{chunk.strip()}'")
        else:
            print("\n\nTiktoken not available. Install with: pip install tiktoken")
    
    def demo_document_specific_splitters(self):
        """Demonstrate document-specific splitters"""
        print("\n" + "=" * 60)
        print("DOCUMENT-SPECIFIC SPLITTERS")
        print("=" * 60)
        
        # 1. Markdown Splitter
        print("\n1. MARKDOWN TEXT SPLITTER")
        print("-" * 30)
        
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        
        markdown_text = self.sample_texts['markdown']
        chunks = markdown_splitter.split_text(markdown_text)
        
        print(f"Number of markdown chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nMarkdown Chunk {i+1} ({len(chunk)} chars):")
            print(f"'{chunk.strip()}'")
            print("-" * 40)
        
        # 2. Python Code Splitter
        print("\n\n2. PYTHON CODE TEXT SPLITTER")
        print("-" * 35)
        
        python_splitter = PythonCodeTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        
        code_text = self.sample_texts['code']
        chunks = python_splitter.split_text(code_text)
        
        print(f"Number of code chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nCode Chunk {i+1} ({len(chunk)} chars):")
            print(f"'{chunk[:200]}...'")  # Show first 200 chars
            print("-" * 50)


class SemanticTextSplitter:
    """Advanced semantic text splitter using embeddings"""
    
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
        )
    
    def split_text_semantically(self, text, embeddings_function=None):
        """Split text based on semantic similarity"""
        
        # First, split into base chunks
        base_chunks = self.base_splitter.split_text(text)
        
        if len(base_chunks) <= 1 or not embeddings_function:
            return base_chunks
        
        # This is a placeholder for semantic splitting
        # In practice, you would use actual embeddings here
        print("Semantic splitting would require actual embeddings model")
        print("Falling back to base chunking...")
        
        return base_chunks


class SlidingWindowSplitter:
    """Sliding window text splitter for better context preservation"""
    
    def __init__(self, window_size=3, step_size=1):
        self.window_size = window_size
        self.step_size = step_size
    
    def split_sentences(self, text):
        """Split text into sentences and create sliding windows"""
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        windows = []
        for i in range(0, len(sentences) - self.window_size + 1, self.step_size):
            window = sentences[i:i + self.window_size]
            windows.append(". ".join(window) + ".")
        
        return windows


class LegalDocumentSplitter(TextSplitter):
    """Custom splitter for legal documents"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Legal document patterns
        self.section_pattern = r'\n\s*(?:SECTION|Section|Article|Chapter)\s+\d+'
        self.subsection_pattern = r'\n\s*\([a-zA-Z0-9]+\)'
        self.paragraph_pattern = r'\n\s*\d+\.'
    
    def split_text(self, text: str) -> List[str]:
        """Split legal text preserving structure"""
        
        # First try to split by sections
        sections = re.split(self.section_pattern, text)
        chunks = []
        
        for section in sections:
            if len(section.strip()) == 0:
                continue
                
            if len(section) <= self._chunk_size:
                chunks.append(section.strip())
            else:
                # Split large sections by subsections
                subsections = re.split(self.subsection_pattern, section)
                for subsection in subsections:
                    if len(subsection.strip()) == 0:
                        continue
                        
                    if len(subsection) <= self._chunk_size:
                        chunks.append(subsection.strip())
                    else:
                        # Further split by paragraphs
                        paragraphs = re.split(self.paragraph_pattern, subsection)
                        for para in paragraphs:
                            if para.strip():
                                chunks.append(para.strip())
        
        return chunks


class ConversationSplitter:
    """Splitter designed for conversational data"""
    
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


class TextSplitterUtils:
    """Utility functions for text splitting optimization and analysis"""
    
    @staticmethod
    def optimize_chunk_size(text, target_model="gpt-3.5-turbo"):
        """Determine optimal chunk size based on content and model"""
        
        model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "claude": 100000,
        }
        
        max_tokens = model_limits.get(target_model, 4096)
        available_tokens = max_tokens - 1000  # Reserve for prompt/response
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_chunk_size = available_tokens * 4
        
        # Adjust based on text characteristics
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 6:  # Technical text
                estimated_chunk_size = int(estimated_chunk_size * 0.8)
        
        return min(estimated_chunk_size, 2000)  # Cap at reasonable size
    
    @staticmethod
    def create_adaptive_splitter(text, target_model="gpt-3.5-turbo"):
        """Create splitter adapted to content and model"""
        
        chunk_size = TextSplitterUtils.optimize_chunk_size(text, target_model)
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
    
    @staticmethod
    def assess_chunk_quality(chunks):
        """Assess the quality of text chunks"""
        
        if not chunks:
            return {"error": "No chunks provided"}
        
        metrics = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks),
            'size_variance': 0,
            'empty_chunks': sum(1 for chunk in chunks if not chunk.strip()),
            'very_short_chunks': sum(1 for chunk in chunks if len(chunk) < 50),
            'very_long_chunks': sum(1 for chunk in chunks if len(chunk) > 2000),
            'min_size': min(len(chunk) for chunk in chunks),
            'max_size': max(len(chunk) for chunk in chunks),
        }
        
        # Calculate size variance
        avg_size = metrics['avg_chunk_size']
        variance = sum((len(chunk) - avg_size) ** 2 for chunk in chunks) / len(chunks)
        metrics['size_variance'] = variance ** 0.5
        
        return metrics
    
    @staticmethod
    def print_chunk_analysis(chunks):
        """Print detailed analysis of chunks"""
        metrics = TextSplitterUtils.assess_chunk_quality(chunks)
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        print("\n" + "=" * 50)
        print("CHUNK QUALITY ANALYSIS")
        print("=" * 50)
        print(f"Total chunks: {metrics['total_chunks']}")
        print(f"Average chunk size: {metrics['avg_chunk_size']:.1f} characters")
        print(f"Size range: {metrics['min_size']} - {metrics['max_size']} characters")
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


def demo_advanced_splitters():
    """Demonstrate advanced and custom splitters"""
    print("\n" + "=" * 60)
    print("ADVANCED AND CUSTOM SPLITTERS")
    print("=" * 60)
    
    examples = TextSplitterExamples()
    
    # 1. Sliding Window Splitter
    print("\n1. SLIDING WINDOW SPLITTER")
    print("-" * 30)
    
    sliding_splitter = SlidingWindowSplitter(window_size=3, step_size=2)
    text = examples.sample_texts['general']
    
    windows = sliding_splitter.split_sentences(text)
    print(f"Number of sliding windows: {len(windows)}")
    
    for i, window in enumerate(windows[:3]):  # Show first 3
        print(f"\nWindow {i+1}:")
        print(f"'{window}'")
    
    # 2. Legal Document Splitter
    print("\n\n2. LEGAL DOCUMENT SPLITTER")
    print("-" * 32)
    
    legal_splitter = LegalDocumentSplitter(chunk_size=300)
    legal_text = examples.sample_texts['legal']
    
    chunks = legal_splitter.split_text(legal_text)
    print(f"Number of legal chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks[:2]):  # Show first 2
        print(f"\nLegal Chunk {i+1}:")
        print(f"'{chunk[:150]}...'")
    
    # 3. Conversation Splitter
    print("\n\n3. CONVERSATION SPLITTER")
    print("-" * 27)
    
    conv_splitter = ConversationSplitter(max_exchanges=3, overlap_exchanges=1)
    conv_text = examples.sample_texts['conversation']
    
    chunks = conv_splitter.split_conversation(conv_text)
    print(f"Number of conversation chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nConversation Chunk {i+1}:")
        print(chunk)
        print("-" * 40)


def demo_best_practices():
    """Demonstrate best practices and utilities"""
    print("\n" + "=" * 60)
    print("BEST PRACTICES AND UTILITIES")
    print("=" * 60)
    
    examples = TextSplitterExamples()
    text = examples.sample_texts['general']
    
    # 1. Adaptive Splitter Selection
    print("\n1. ADAPTIVE SPLITTER SELECTION")
    print("-" * 35)
    
    for model in ["gpt-3.5-turbo", "gpt-4", "claude"]:
        optimal_size = TextSplitterUtils.optimize_chunk_size(text, model)
        print(f"{model}: Optimal chunk size = {optimal_size} chars")
    
    # Create adaptive splitter
    adaptive_splitter = TextSplitterUtils.create_adaptive_splitter(text, "gpt-4")
    chunks = adaptive_splitter.split_text(text)
    
    print(f"\nAdaptive splitter created {len(chunks)} chunks")
    
    # 2. Quality Analysis
    print("\n\n2. CHUNK QUALITY ANALYSIS")
    print("-" * 30)
    
    # Compare different splitters
    splitters = {
        "Character": CharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        "Recursive": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        "Token": TokenTextSplitter(chunk_size=50, chunk_overlap=10),
    }
    
    for name, splitter in splitters.items():
        print(f"\n--- {name} Splitter ---")
        chunks = splitter.split_text(text)
        TextSplitterUtils.print_chunk_analysis(chunks)


def main():
    """Main function to run all demonstrations"""
    print("LANGCHAIN TEXT SPLITTERS DEMONSTRATION")
    print("=" * 60)
    print("Complete guide from basics to advanced techniques")
    print("=" * 60)
    
    # Initialize examples
    examples = TextSplitterExamples()
    
    try:
        # Run all demonstrations
        examples.demo_basic_splitters()
        examples.demo_character_based_splitters()
        examples.demo_token_based_splitters()
        examples.demo_document_specific_splitters()
        
        demo