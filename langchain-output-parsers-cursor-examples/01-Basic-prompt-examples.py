from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate   
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts.few_shot import FewShotPromptTemplate  # <-- Add this import
from langchain_core.example_selectors import LengthBasedExampleSelector  # <-- Add this import
# Initialize the Google Generative AI client
from dotenv import load_dotenv
load_dotenv()
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
""" 
# Define a simple prompt
prompt = "What is the capital of France?"
# Generate a response
response = model.invoke(prompt)
# Print the response
print("Response:", response.content)
"""

def simple_prompt_example():
    # Define a simple prompt
    prompt = "Explain the concept of machine learning in simple terms."
    # Generate a response
    response = model.invoke(prompt)
    # Print the response
    print("Response:", response.content)


def template_prompt_example():
    # Define a test prompt
    template ="""
    You are a helpful assistant that answers questions about technical questions.
    please explain {concept} in a way that {audience} can understand.
    keep your ansswr under {word_limit} words.
    
    """

    prompt_template = PromptTemplate(
        input_variables=["concept", "audience", "word_limit"],
        template=template
    )

    # Create a prompt from the template
    prompt = prompt_template.format(
        concept="machine learning",
        audience="a beginner",
        word_limit=50
    )

    print("Prompt:", prompt)
    # Generate a response
    response = model.invoke(prompt)
    # Print the response
    print("Response:", response.content)


def template_prompt_example_with_hf():

    # Initialize the Hugging Face model TinyLlama/TinyLlama-1.1B-Chat-v1.0
   # Initialize the Hugging Face chat model
    llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
                                                     task="text-generation")
    chat_model = ChatHuggingFace(llm=llm)

    # Define a test prompt
    template = """
            You are a professional product marketer.
            Write a compelling product description based on the following details:

            Product Name: {product_name}
            Key Features: {features}

            Product Description:
            """
   
    # Create a prompt from the template
    prompt_template = PromptTemplate(
        input_variables=["product_name", "features"],
        template=template.strip()
    )
    prompt = prompt_template.format(
        product_name="Smartphone X",
        features="5G connectivity, 128GB storage, 48MP camera"
    )

    print("Prompt:", prompt)
    # Generate a response
    response = chat_model.invoke(prompt)
    # Print the response
    print("Response:", response.content)

def simple_chat_prompt_example():
    # Define a simple chat prompt
    template = """
    You are a helpful assistant.
    User: {user_input}
    Assistant:
    """

    chat_prompt_template = ChatPromptTemplate.from_template(template)

    # Create a chat prompt from the template
    chat_prompt = chat_prompt_template.format(user_input="What is the capital of France?")

    print("Chat Prompt:", chat_prompt)
    # Generate a response
    response = model.invoke(chat_prompt)
    # Print the response
    print("Response:", response.content)


def  chat_template_prompt_example():
    # Define a template for a chat prompt
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a coding tutor who helps students understand programming concepts."),
        ("user", "Explain {concept} with a practical example in {language}."),
        ("assistant", "")
    ])
    # Create a chat prompt from the template
    chat_prompt = chat_template.format(
        concept="object-oriented programming",
        language="Python"
    )
    print("Chat Prompt:", chat_prompt)
    # Generate a response
    response = model.invoke(chat_prompt)
    # Print the response
    print("Response:", response.content)


def chat_template_prompt_example_with_hf():
    # Initialize the Hugging Face chat model
    #llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
    #                         task="text-generation")
    #chat_model = ChatHuggingFace(llm=llm)

    # Define a template for a chat prompt
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel planner. Always respond with friendly and helpful travel advice."),
        ("user", "I'm planning a {days}-day trip to {destination} in {month}. I enjoy {interests}. Can you suggest an itinerary?"),
        ("assistant", "Sure! Here's a suggested itinerary for your trip:")
    ])
    # Create a chat prompt from the template

    messages = chat_template.format_messages(
        days=7,
        destination="Pune India",
        month="June",
        interests="historical sites and local cuisine"
    )

    for message in messages:
        print(f"{message.type}: {message.content}")

    # Generate a response
    response = model.invoke(messages)
    # Print the response
    print("Response:", response.content)


def few_shot_chat_template_prompt_example():
    
    # sentiment analysis example
    # Define a template for a few-shot chat prompt
    examples = [
        {"txt": "I love this product!", "sentiment": "positive"},
        {"txt": "This is the worst experience I've ever had.", "sentiment": "negative"},
        {"txt": "It's okay, not great but not bad either.", "sentiment": "neutral"},
        {"txt": "The service was excellent and the food was delicious!", "sentiment": "positive"},
        {"txt": "I am disappointed with the quality of this item.", "sentiment": "negative"},
        {"txt": "I had a great time at the event, it was well organized.", "sentiment": "positive"},
        {"txt": "The product arrived late and was damaged.", "sentiment": "negative"},
        {"txt": "I think the service could be improved.", "sentiment": "neutral"},
        {"txt": "Absolutely fantastic! I will buy again.", "sentiment": "positive"},
        {"txt": "Not worth the money, very dissatisfied.", "sentiment": "negative"}

    ]

    example_prompts = PromptTemplate(

        input_variables=["txt", "sentiment"],
        template="Text: {txt}\nSentiment: {sentiment}"

    )

    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompts,
        input_variables=["input_text"],
        prefix="Classify the sentiment of the following texts as positive, negative, or neutral.",
        suffix="Text: {input_text}\nSentiment:",
        example_separator="\n\n",
    )

    test_text = "The movie was fantastic!"
    # Create a few-shot prompt
    prompt = few_shot_template.format(input_text=test_text)
    print("Few-shot Prompt:\n", prompt)
    # Generate a response
    response = model.invoke(prompt)
    # Print the response
    print("Response:", response.content)


def dynamic_few_shot_prompt_template_example():
    from langchain.prompts.example_selector import LengthBasedExampleSelector
    from langchain.prompts.prompt import PromptTemplate
    from langchain.prompts.few_shot import FewShotPromptTemplate
    # Make sure to import or define your 'model' before using it

    examples = [
        {"input": "2 + 2", "output": "4"},
        {"input": "5 * 3", "output": "15"},
        {"input": "10 / 2", "output": "5"},
        {"input": "7 - 4", "output": "3"},
        {"input": "3^2", "output": "9"},
        {"input": "sqrt(16)", "output": "4"},
        {"input": "15 % 4", "output": "3"},
        {"input": "2^3", "output": "8"},
        {"input": "20 / 5", "output": "4"},
        {"input": "6 + 9", "output": "15"}
    ]

    # Define a template for the few-shot prompt
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    )

    # Length based example selection (example_prompt is required)
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,  # <-- required argument
        max_length=200,
        get_text_length=lambda x: len(str(x))
    )

    few_shot_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        input_variables=["input_text"],
        prefix="Solve the following mathematical expressions.",
        suffix="Input: {input_text}\nOutput:",
        example_separator="\n\n"
    )

    test_input = ["8+3", "12 / 4", "5 * 6" , "9 - 2"]

    for input_text in test_input:
        # Create a few-shot prompt
        prompt = few_shot_template.format(input_text=input_text)
        print("Few-shot Prompt:\n", prompt)
        # Generate a response
        response = model.invoke(prompt)
        # Print the response
        print(f"Response for '{input_text}':", response.content)




    # Example usage (uncomment to test)
    # prompt = few_shot_template.format(txt="The movie was fantastic!")
    # print("Few-shot Prompt:\n", prompt)

#simple_prompt_example()
#simple_prompt_example()

#template_prompt_example()

#template_prompt_example_with_hf()

#simple_chat_prompt_example()

#chat_template_prompt_example()

#chat_template_prompt_example_with_hf()


#few_shot_chat_template_prompt_example()


dynamic_few_shot_prompt_template_example()
