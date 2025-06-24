# LangChain Chain Examples

This project demonstrates multiple LangChain chain patterns, from basic to advanced, using the OpenAI LLM.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or, if using pyproject.toml:
   pip install .
   ```
2. **Set your OpenAI API key:**
   - Set the environment variable `OPENAI_API_KEY` with your OpenAI API key:
     - On Windows (PowerShell):
       ```powershell
       $env:OPENAI_API_KEY="your-api-key-here"
       ```
     - On Linux/macOS:
       ```bash
       export OPENAI_API_KEY="your-api-key-here"
       ```

## Running the Examples

Each chain type is in its own file. Run any example with:
```bash
python <filename>.py
```

### Files and Examples

- **simple_chain.py**: Basic LLMChain (single prompt, single output)
- **simple_sequential_chain.py**: SimpleSequentialChain (output of one chain is input to the next)
- **sequential_chain.py**: SequentialChain (multi-input/output, advanced chaining)
- **parallel_chain.py**: Simulated parallel chains (run two chains independently and combine results)
- **condition_chain.py**: Conditional chain (choose which chain to run based on input)

Each script prints its results to the console.
