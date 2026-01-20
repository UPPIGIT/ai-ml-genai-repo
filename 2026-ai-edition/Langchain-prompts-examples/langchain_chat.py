# 02_chat_prompt_templates.py
# ChatPromptTemplate is used for chat-based models (like GPT, Claude)
# It structures conversations with system, human, and AI messages

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

# Example 1: Basic ChatPromptTemplate with system and human messages
# System message sets the AI's behavior/role
# Human message is the user's input
chat_template1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

# Format the entire conversation
messages1 = chat_template1.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming"
)

print("Example 1 - Basic Chat Template:")
for msg in messages1:
    print(f"{msg.type}: {msg.content}")
print("\n" + "="*50 + "\n")

# Example 2: Multi-turn conversation template
# This simulates a conversation with multiple exchanges
chat_template2 = ChatPromptTemplate.from_messages([
    ("system", "You are a coding tutor. Be patient and explain clearly."),
    ("human", "What is a variable?"),
    ("ai", "A variable is a container for storing data values."),
    ("human", "Can you give me an example in {language}?")
])

messages2 = chat_template2.format_messages(language="Python")

print("Example 2 - Multi-turn Conversation:")
for msg in messages2:
    print(f"{msg.type}: {msg.content}")
print("\n" + "="*50 + "\n")

# Example 3: Using MessagePromptTemplate classes for more control
# This gives you finer control over each message type
system_template = SystemMessagePromptTemplate.from_template(
    "You are an expert {role}. Your expertise is in {domain}."
)

human_template = HumanMessagePromptTemplate.from_template(
    "{user_question}"
)

# Combine them into a chat template
chat_template3 = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

messages3 = chat_template3.format_messages(
    role="data scientist",
    domain="machine learning",
    user_question="What is overfitting?"
)

print("Example 3 - Structured Message Templates:")
for msg in messages3:
    print(f"{msg.type}: {msg.content}")
print("\n" + "="*50 + "\n")

# Example 4: Template with multiple human-AI exchanges
# Useful for few-shot learning (providing examples)
chat_template4 = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer. Respond only with: Positive, Negative, or Neutral"),
    ("human", "I love this product!"),
    ("ai", "Positive"),
    ("human", "This is terrible."),
    ("ai", "Negative"),
    ("human", "{text_to_analyze}")
])

messages4 = chat_template4.format_messages(
    text_to_analyze="It's okay, nothing special."
)

print("Example 4 - Few-Shot Learning Template:")
for msg in messages4:
    print(f"{msg.type}: {msg.content}")