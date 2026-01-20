# 03_partial_prompts.py
# Partial prompts allow you to "pre-fill" some variables while leaving others open
# This is useful when you know some values ahead of time but not others

from langchain.prompts import PromptTemplate
from datetime import datetime
from functools import partial

# Example 1: Basic partial with static values
# Pre-fill the 'style' variable, leave 'topic' for later
base_template = PromptTemplate(
    input_variables=["style", "topic"],
    template="Write a {style} article about {topic}"
)

# Create a partial template with 'style' already filled in
technical_template = base_template.partial(style="technical")

# Now we only need to provide 'topic'
prompt1 = technical_template.format(topic="cloud computing")
print("Example 1 - Static Partial:")
print(prompt1)
print("\n" + "="*50 + "\n")

# Example 2: Partial with function (dynamic values)
# This is powerful when you want values calculated at format time
# For example, always including the current date

def get_current_date():
    """Returns today's date in a readable format"""
    return datetime.now().strftime("%B %d, %Y")

date_template = PromptTemplate(
    input_variables=["task"],
    template="Today is {date}. {task}",
    # Use partial_variables for functions
    partial_variables={"date": get_current_date}
)

prompt2 = date_template.format(task="Summarize the news")
print("Example 2 - Dynamic Partial with Function:")
print(prompt2)
print("\n" + "="*50 + "\n")

# Example 3: Multiple partials for reusable templates
# Create a base template for customer emails
email_template = PromptTemplate(
    input_variables=["customer_name", "product", "issue", "tone"],
    template="""
Dear {customer_name},

We understand you're experiencing an issue with {product}: {issue}

{tone}

Best regards,
Support Team
"""
)

# Create different versions with pre-filled tones
friendly_email = email_template.partial(
    tone="We're here to help! Let's get this sorted out for you right away."
)

formal_email = email_template.partial(
    tone="We apologize for the inconvenience and will address this matter promptly."
)

# Use the friendly version
prompt3 = friendly_email.format(
    customer_name="John",
    product="Widget Pro",
    issue="unable to login"
)

print("Example 3 - Multiple Partial Variants:")
print(prompt3)
print("\n" + "="*50 + "\n")

# Example 4: Chaining partials
# You can apply partial multiple times
report_template = PromptTemplate(
    input_variables=["department", "metric", "period", "format"],
    template="Generate a {format} report for {department} showing {metric} for {period}"
)

# First partial: set the format
formatted_report = report_template.partial(format="detailed quarterly")

# Second partial: set the department
sales_report = formatted_report.partial(department="Sales Department")

# Now only need metric and period
prompt4 = sales_report.format(metric="revenue growth", period="Q4 2024")

print("Example 4 - Chained Partials:")
print(prompt4)
print("\n" + "="*50 + "\n")

# Example 5: Practical use case - API context
# Pre-fill API or system context that shouldn't change
def get_api_version():
    return "v2.1"

def get_user_tier():
    return "Premium"

api_prompt = PromptTemplate(
    input_variables=["query"],
    template="[API {api_version}] [User: {user_tier}] Process: {query}",
    partial_variables={
        "api_version": get_api_version,
        "user_tier": get_user_tier
    }
)

prompt5 = api_prompt.format(query="fetch user analytics")
print("Example 5 - API Context with Functions:")
print(prompt5)