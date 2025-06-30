# LangChain Runnable Examples

This repository demonstrates the core concepts and advanced usage of **Runnables** in [LangChain](https://python.langchain.com/). Each example is self-contained and well-commented, making it easy to learn and experiment.

---

## What is a Runnable?

A **Runnable** in LangChain is any component that can be invoked with an input and returns an output. Runnables are the building blocks for building flexible, composable, and powerful data and LLM pipelines.

**Key Points:**
- Runnables can be models, chains, tools, or custom logic.
- They can be composed using operators (like `|` for chaining) or special classes (like `RunnableParallel`, `RunnableBranch`).
- Runnables support both synchronous (`invoke`) and streaming (`stream`) execution.
- They enable modular, reusable, and testable code for LLM and data workflows.

---

## Core Runnable Types

- **RunnableLambda**: Wraps a Python function as a runnable. Great for custom logic, post-processing, or data transformation.
- **RunnableParallel**: Runs multiple runnables in parallel and collects their outputs in a dictionary.
- **RunnablePassthrough**: Returns its input unchanged. Useful for debugging, logging, or as a placeholder in chains.
- **RunnableSequence**: Chains runnables in a specific order, passing the output of one as the input to the next.
- **RunnableBranch**: Routes input to different runnables based on conditions (like an if-elif-else for pipelines).

---

## Example Files

### Basic and Custom Examples
- `01_basic_runnable.py`: Basic runnable chain and streaming output.
- `02_custom_runnable.py`: Custom runnable functions and classes.
- `03_conditional_runnable.py`: Conditional logic and branching in chains.

### Runnable Type Demos
- `04_runnable_types_examples.py`: One file with simple examples for all core types.
- `05_runnable_lambda_examples.py`: Multiple `RunnableLambda` examples.
- `06_runnable_parallel_examples.py`: Multiple `RunnableParallel` examples.
- `07_runnable_passthrough_examples.py`: Multiple `RunnablePassthrough` examples.
- `08_runnable_sequence_examples.py`: Multiple `RunnableSequence` examples.
- `09_runnable_branch_examples.py`: Multiple `RunnableBranch` examples.

### Advanced/Real-World Examples
- `10_runnable_lambda_advanced.py`: Text cleaning and normalization pipeline.
- `11_runnable_parallel_advanced.py`: Parallel text feature extraction (sentiment, word count, language detection).
- `12_runnable_passthrough_advanced.py`: Logging and conditional data forwarding.
- `13_runnable_sequence_advanced.py`: Multi-step data enrichment (clean, extract, summarize).
- `14_runnable_branch_advanced.py`: Dynamic workflow routing (e.g., support ticket triage).

### LLM-Integrated Examples
- `15_runnable_lambda_llm.py`: LLM output post-processing (e.g., extract keywords).
- `16_runnable_parallel_llm.py`: LLM summary, sentiment, and word count in parallel.
- `17_runnable_passthrough_llm.py`: Logging/enriching LLM input/output.
- `18_runnable_sequence_llm.py`: Multi-step LLM pipeline (prompt → LLM → post-process → LLM follow-up).
- `19_runnable_branch_llm.py`: Route input to different LLM prompts based on user intent.

---

## How to Use

1. **Install dependencies:**
   ```bash
   pip install langchain langchain-openai
   ```
2. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY=your-key-here
   # or on Windows:
   set OPENAI_API_KEY=your-key-here
   ```
3. **Run any example:**
   ```bash
   python 05_runnable_lambda_examples.py
   # or any other file
   ```

---

## When to Use Each Runnable Type

- **RunnableLambda**: For custom logic, data cleaning, or post-processing.
- **RunnableParallel**: When you want to extract or compute multiple features at once.
- **RunnablePassthrough**: For debugging, logging, or as a placeholder in a chain.
- **RunnableSequence**: For multi-step workflows where each step depends on the previous.
- **RunnableBranch**: For dynamic routing, conditional logic, or workflow automation.

---

## More Resources
- [LangChain Python Docs](https://python.langchain.com/)
- [LangChain Runnables Guide](https://python.langchain.com/docs/expression_language/runnables/)

---

**Happy chaining!** 