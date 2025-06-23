# Setting Up Open Source Models for LangChain Prompts

This guide will help you set up and use open source models with the LangChain prompt examples.

## Overview

The examples now support multiple open source model providers:

1. **Ollama** - Easy-to-use local model runner
2. **Hugging Face** - Large collection of open source models
3. **Local Models** - Models running on your own hardware

## Prerequisites

### 1. Install uv (if not already done)
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Project Dependencies
```bash
# Install core dependencies
uv sync

# Install with specific model support
uv sync --extra ollama
uv sync --extra huggingface
uv sync --extra local
```

## Option 1: Ollama Setup (Recommended for Beginners)

Ollama is the easiest way to run open source models locally.

### 1. Install Ollama
```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama Service
```bash
ollama serve
```

### 3. Download Models
```bash
# Download Llama 2 (7B parameters)
ollama pull llama2

# Download Mistral (7B parameters)
ollama pull mistral

# Download Code Llama (for programming tasks)
ollama pull codellama

# Download Phi-2 (Microsoft's small model)
ollama pull phi
```

### 4. Test Ollama
```bash
# Test a model
ollama run llama2 "Hello, how are you?"
```

### 5. Run Examples with Ollama
```bash
# Run basic examples
uv run python basic_prompts_opensource.py

# Run advanced examples
uv run python advanced_prompts_opensource.py
```

## Option 2: Hugging Face Setup

Hugging Face provides access to thousands of open source models.

### 1. Install Hugging Face Dependencies
```bash
uv sync --extra huggingface
```

### 2. Set Up Hugging Face Hub (Optional)
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (optional, for private models)
huggingface-cli login
```

### 3. Available Models

The examples use these models by default:

- **microsoft/DialoGPT-medium** - Good for conversations
- **gpt2** - OpenAI's GPT-2 model
- **EleutherAI/gpt-neo-125M** - Smaller, faster model
- **microsoft/DialoGPT-small** - Smallest, fastest model

### 4. Run Examples with Hugging Face
```bash
# Models will be downloaded automatically on first use
uv run python basic_prompts_opensource.py
```

## Option 3: Local Model Setup

For advanced users who want to run models directly on their hardware.

### 1. Install Local Dependencies
```bash
uv sync --extra local
```

### 2. Download Model Files
```bash
# Example: Download a GGML model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### 3. Configure Local Models
Edit the `model_setup.py` file to add your local model paths.

## Model Comparison

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **Llama 2** | 7B | Medium | High | General purpose |
| **Mistral** | 7B | Fast | High | General purpose |
| **Code Llama** | 7B | Medium | High | Programming |
| **Phi-2** | 2.7B | Very Fast | Good | Lightweight tasks |
| **DialoGPT-medium** | 345M | Fast | Good | Conversations |
| **GPT-2** | 124M | Very Fast | Medium | Text generation |

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 8GB RAM
- **Storage**: 10GB free space
- **Network**: Internet connection for model downloads

### Recommended Requirements
- **CPU**: 8+ cores, 16GB+ RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for faster inference)
- **Storage**: 50GB+ free space
- **Network**: Fast internet for model downloads

### GPU Setup (Optional)
```bash
# Install CUDA support for PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install GPU support for transformers
pip install accelerate
```

## Configuration

### Environment Variables
Create a `.env` file with your preferences:

```bash
# Model preferences
DEFAULT_MODEL_TYPE=ollama
DEFAULT_MODEL_NAME=llama2

# Hugging Face settings
HUGGINGFACE_CACHE_DIR=./models
HUGGINGFACE_OFFLINE=0

# Ollama settings
OLLAMA_HOST=http://localhost:11434
```

### Model Configuration
Edit `model_setup.py` to customize model settings:

```python
# Add custom model configurations
MODEL_CONFIGS["ollama"]["custom_model"] = {
    "name": "custom_model",
    "description": "My custom model",
    "temperature": 0.5,
    "max_tokens": 1024
}
```

## Troubleshooting

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
   
   # Retry model download
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')"
   ```

3. **Memory Issues**:
   - Use smaller models (Phi-2, DialoGPT-small)
   - Reduce batch size in model configuration
   - Close other applications to free memory

4. **GPU Issues**:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   ```

### Performance Optimization

1. **For CPU-only systems**:
   - Use smaller models (Phi-2, DialoGPT-small)
   - Reduce max_tokens in prompts
   - Use quantized models when available

2. **For GPU systems**:
   - Enable CUDA support
   - Use larger models for better quality
   - Adjust batch sizes for optimal performance

3. **For production use**:
   - Use model serving frameworks (vLLM, Text Generation Inference)
   - Implement caching for repeated prompts
   - Monitor resource usage

## Testing Your Setup

### 1. Check Model Availability
```bash
python model_setup.py
```

### 2. Run Basic Test
```bash
python -c "
from model_setup import get_model
llm = get_model('ollama', 'llama2')
print(llm.invoke('Hello, world!'))
"
```

### 3. Run Full Examples
```bash
# Run all examples
uv run python run_all_examples.py

# Run specific examples
uv run python basic_prompts_opensource.py
uv run python advanced_prompts_opensource.py
```

## Next Steps

1. **Experiment with different models**:
   - Try different Ollama models
   - Test Hugging Face models
   - Compare performance and quality

2. **Customize prompts**:
   - Modify examples for your use case
   - Create new prompt templates
   - Optimize for your specific model

3. **Scale up**:
   - Set up model serving
   - Implement caching
   - Add monitoring and logging

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Hugging Face Models](https://huggingface.co/models)
- [LangChain Documentation](https://python.langchain.com/)
- [Model Performance Benchmarks](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your model installation
3. Check system requirements
4. Review error messages carefully
5. Try with a different model

For additional help, refer to the model-specific documentation or community forums. 