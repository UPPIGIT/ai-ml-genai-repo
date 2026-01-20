# 01_basic_prompt_template.py
# This demonstrates the most basic use of LangChain prompt templates
# A PromptTemplate allows you to create reusable prompts with variables

from langchain.prompts import PromptTemplate

# Example 1: Simple string template with one variable
# The {topic} is a placeholder that will be replaced with actual values
simple_template = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a joke about {topic}"
)

# Format the prompt by providing the variable value
prompt1 = simple_template.format(topic="programming")
print("Example 1 - Simple Template:")
print(prompt1)
print("\n" + "="*50 + "\n")

# Example 2: Template with multiple variables
# You can have multiple placeholders in your template
multi_var_template = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Write a {adjective} story about {content}"
)

prompt2 = multi_var_template.format(adjective="scary", content="a haunted house")
print("Example 2 - Multiple Variables:")
print(prompt2)
print("\n" + "="*50 + "\n")

# Example 3: Using from_template() - a shorter syntax
# This automatically detects variables in curly braces
quick_template = PromptTemplate.from_template(
    "Translate the following text to {language}: {text}"
)

prompt3 = quick_template.format(language="Spanish", text="Hello, how are you?")
print("Example 3 - Quick Template Creation:")
print(prompt3)
print("\n" + "="*50 + "\n")

# Example 4: Template with no variables
# Sometimes you just want a static prompt
static_template = PromptTemplate(
    input_variables=[],
    template="List 5 programming best practices"
)

prompt4 = static_template.format()
print("Example 4 - Static Template:")
print(prompt4)