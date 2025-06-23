# LangChain Prompt Examples with Open Source Models

This repository contains comprehensive examples of LangChain prompts using **open source models** instead of commercial APIs. Learn to build AI applications with free, locally-run models.

## 🌟 Features

- **Open Source Models**: Use Ollama, Hugging Face, and local models
- **No API Costs**: Run everything locally without usage fees
- **Comprehensive Examples**: From basic prompts to advanced techniques
- **Multiple Model Support**: Llama 2, Mistral, Code Llama, Phi-2, and more
- **Easy Setup**: Simple installation with uv package manager

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
uv sync --extra ollama
uv sync --extra huggingface
```

### 2. Set Up Ollama (Recommended)
```bash
# Install Ollama
brew install ollama  # macOS
# or curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start Ollama and download models
ollama serve
ollama pull llama2
ollama pull mistral
```

### 3. Configure Environment
```bash
cp env_example.txt .env
# Edit .env with your preferences (optional)
```

### 4. Run Examples
```bash
# Run all open source examples
uv run python run_opensource_examples.py

# Or run individual examples
uv run python basic_prompts_opensource.py
uv run python advanced_prompts_opensource.py
```

## 📚 Examples Included

### 1. Basic Prompts (`basic_prompts_opensource.py`)
- Simple text prompts with open source models
- Template prompts with variables
- Few-shot learning examples
- Dynamic example selection

### 2. Advanced Prompts (`advanced_prompts_opensource.py`)
- Chain of thought prompting
- Role-based prompts
- Structured output parsing
- Multi-step reasoning
- Conditional prompting
- Creative prompting techniques

### 3. Model Setup (`model_setup.py`)
- Model management and configuration
- Multiple model provider support
- Embeddings setup
- Model comparison utilities

## 🤖 Supported Models

### Ollama Models (Recommended)
- **Llama 2** - Meta's 7B parameter model
- **Mistral** - Mistral AI's 7B model
- **Code Llama** - Specialized for programming
- **Phi-2** - Microsoft's lightweight 2.7B model

### Hugging Face Models
- **microsoft/DialoGPT-medium** - Conversational AI
- **gpt2** - OpenAI's GPT-2 model
- **EleutherAI/gpt-neo-125M** - Lightweight model
- **microsoft/DialoGPT-small** - Fastest option

## 🛠️ Setup Options

### Option 1: Ollama (Recommended for Beginners)
```bash
# Install Ollama
brew install ollama

# Download models
ollama pull llama2
ollama pull mistral

# Run examples
uv run python basic_prompts_opensource.py
```

### Option 2: Hugging Face
```bash
# Install dependencies
uv sync --extra huggingface

# Models download automatically on first use
uv run python basic_prompts_opensource.py
```

### Option 3: Local Models
```bash
# Install local model support
uv sync --extra local

# Configure your own model files
# Edit model_setup.py to add custom models
```

## 💻 Usage Examples

### Basic Prompt with Ollama
```python
from model_setup import get_model

# Get Llama 2 model
llm = get_model("ollama", "llama2")

# Simple prompt
response = llm.invoke("Explain quantum computing in simple terms.")
print(response)
```

### Advanced Prompt with Hugging Face
```python
from model_setup import get_model
from langchain.prompts import PromptTemplate

# Get Hugging Face model
llm = get_model("huggingface", "microsoft/DialoGPT-medium")

# Template prompt
template = "Explain {concept} to a {audience}."
prompt = PromptTemplate(input_variables=["concept", "audience"], template=template)
formatted = prompt.format(concept="machine learning", audience="high school student")

response = llm.invoke(formatted)
print(response)
```

### Model Comparison
```python
from model_setup import model_comparison_example

# Compare different models on the same prompt
model_comparison_example()
```

## 🔧 Configuration

### Environment Variables
```bash
# Model preferences
DEFAULT_MODEL_TYPE=ollama
DEFAULT_MODEL_NAME=llama2

# Ollama settings
OLLAMA_HOST=http://localhost:11434

# Hugging Face settings
HUGGINGFACE_CACHE_DIR=./models
```

### Model Settings
```python
# Customize model parameters
llm = get_model("ollama", "llama2", temperature=0.5, max_tokens=1024)
```

## 📊 Performance Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **Llama 2** | 7B | Medium | High | General purpose |
| **Mistral** | 7B | Fast | High | General purpose |
| **Code Llama** | 7B | Medium | High | Programming |
| **Phi-2** | 2.7B | Very Fast | Good | Lightweight tasks |
| **DialoGPT-medium** | 345M | Fast | Good | Conversations |

## 🚀 Advanced Features

### Chain of Thought Prompting
```python
cot_prompt = """
Let's approach this step by step:
Problem: {problem}
Let me think through this:
1) First, I need to understand what's being asked
2) Then, I'll break it down into smaller parts
3) I'll solve each part systematically
4) Finally, I'll combine the results
"""

response = llm.invoke(cot_prompt.format(problem="Complex math problem"))
```

### Structured Output Parsing
```python
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class BookAnalysis(BaseModel):
    title: str = Field(description="Book title")
    rating: float = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary")

parser = PydanticOutputParser(pydantic_object=BookAnalysis)
response = llm.invoke(f"Analyze this book: {book_description}\n{parser.get_format_instructions()}")
parsed = parser.parse(response)
```

## 🛠️ Development

### Code Quality Tools
```bash
# Install development dependencies
uv sync --extra dev

# Format and lint code
uv run black .
uv run isort .
uv run flake8 .

# Type checking
uv run mypy .
```

### Adding New Models
```python
# Add custom model configuration
MODEL_CONFIGS["ollama"]["custom_model"] = {
    "name": "custom_model",
    "description": "My custom model",
    "temperature": 0.5,
    "max_tokens": 1024
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama service
   ollama serve
   ```

2. **Model Download Issues**:
   ```bash
   # Clear Hugging Face cache
   rm -rf ~/.cache/huggingface/
   
   # Check internet connection
   curl -I https://huggingface.co
   ```

3. **Memory Issues**:
   - Use smaller models (Phi-2, DialoGPT-small)
   - Close other applications
   - Reduce batch size

4. **Slow Performance**:
   - Use GPU acceleration if available
   - Try smaller models
   - Optimize prompt length

### Performance Tips

- **For CPU-only systems**: Use Phi-2 or DialoGPT-small
- **For GPU systems**: Enable CUDA support for faster inference
- **For production**: Use model serving frameworks like vLLM

## 📖 Documentation

- [Setup Guide](setup_opensource_models.md) - Detailed setup instructions
- [uv Setup Guide](uv_setup.md) - Package management with uv
- [Model Configurations](model_setup.py) - Available models and settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your examples or improvements
4. Run code quality checks: `uv run black . && uv run isort . && uv run flake8 .`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for easy local model deployment
- [Hugging Face](https://huggingface.co/) for the model hub
- [LangChain](https://python.langchain.com/) for the framework
- [Meta](https://ai.meta.com/) for Llama 2
- [Mistral AI](https://mistral.ai/) for Mistral models

## 🔗 Links

- [Ollama Documentation](https://ollama.ai/docs)
- [Hugging Face Models](https://huggingface.co/models)
- [LangChain Documentation](https://python.langchain.com/)
- [Model Performance Benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 