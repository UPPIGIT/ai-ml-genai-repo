from langchain_core.prompts import PromptTemplate

# Email Rewriting Assistant
EMAIL_REWRITE_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
Rewrite the following email in a {tone} tone.{instructions}

Email:
{email_content}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are an expert email writer. Please rewrite the email below in a {tone} tone. Be sure to improve clarity and professionalism. If additional instructions are provided, follow them closely.

Email:
{email_content}
{instructions}
"""
    ),
}

# Markdown Content Generator
MARKDOWN_GEN_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
Convert the following content into a well-structured Markdown file.{instructions}

Content:
{content}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are a technical writer. Convert the content below into a Markdown document with clear sections, bullet points, and examples if possible. Follow any extra instructions provided.

Content:
{content}
{instructions}
"""
    ),
}

# Question-Answer Assistant - Evaluation
QA_EVAL_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
Given the following question and options, identify the correct answer with explanation, and explain why the other options are incorrect.{instructions}

Question:
{question}
Options:
{options}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are an expert educator. For the question and options below, select the best answer, explain your reasoning, and briefly critique each incorrect option. Follow any extra instructions provided.

Question:
{question}
Options:
{options}
{instructions}
"""
    ),
    'v3': PromptTemplate.from_template(
        """
You are a helpful assistant. Answer the following question in a clear, concise, and conversational way. If there are options, pick the best one and explain briefly. If not, just answer directly. Be friendly and to the point. Follow any extra instructions provided.

Question:
{question}
Options:
{options}
{instructions}
"""
    ),
}

# Question-Answer Assistant - Generation
QA_GEN_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
Generate {num_questions} multiple-choice questions with answers based on the following theory block.{instructions}

Theory Block:
{theory}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are a subject matter expert. Create {num_questions} challenging multiple-choice questions (with answers and explanations) from the theory block below. Use any extra instructions provided.

Theory Block:
{theory}
{instructions}
"""
    ),
}

# Effort Estimation Generator
EFFORT_ESTIMATE_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
Given the following activities and categories, select relevant tasks for the project '{task_name}' and distribute {total_hours} hours among them. Ensure the total matches exactly.{instructions}

CSV Activities:
{csv_activities}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are a project manager. Review the activities below and allocate {total_hours} hours for the project '{task_name}'. Provide a brief justification for each allocation. Follow any extra instructions provided.

CSV Activities:
{csv_activities}
{instructions}
"""
    ),
    'v3': PromptTemplate.from_template(
        """
You are an expert project manager. Given the main story/task: '{task_name}', a list of activities and categories, and a total of {total_hours} hours to distribute:
- Select only the most essential and relevant activities.
- Distribute exactly {total_hours} hours among the selected activities.
- Assign only integer values for hours, and each activity must get between 1 and 40 hours (inclusive).
- Output only a CSV table with header: Category,Activity,Hours (no extra text, no explanation, no blank lines).
- Do not use any other column names. Do not use '...' or blanks. If you cannot estimate hours for an activity, omit it.
- The sum of Hours must be exactly {total_hours}.

Example:
Category,Activity,Hours
Development,Code development,20
QA,Test cases,10
Documentation,Write docs,10

Activities and Categories:
{csv_activities}
{instructions}
""")
}

ASK_ANYTHING_PROMPTS = {
    'v1': PromptTemplate.from_template(
        """
You are a helpful AI assistant. Answer the user's question in a simple, crisp, and conversational way. Be direct and friendly. If the user asks for details, provide them, but otherwise keep it short and clear.

Question:
{question}
"""
    ),
    'v2': PromptTemplate.from_template(
        """
You are a friendly and knowledgeable AI assistant. Give a detailed, helpful, and encouraging answer to the user's question. If the user asks for a summary, keep it short; otherwise, be thorough and supportive.

Question:
{question}
"""
    ),
    'v3': PromptTemplate.from_template(
        """
You are a helpful AI assistant. If the user's question can be answered as a list (steps, tips, items), provide a clear, well-formatted list. Otherwise, answer simply and directly.

Question:
{question}
"""
    ),
}

PROMPT_VERSIONS = {
    'email': {'v1': 'Simple rewrite', 'v2': 'Expert rewrite'},
    'markdown': {'v1': 'Basic markdown', 'v2': 'Technical writer'},
    'qa_eval': {'v1': 'Basic evaluation', 'v2': 'Expert educator', 'v3': 'Conversational, concise answer'},
    'qa_gen': {'v1': 'Basic generation', 'v2': 'Challenging questions'},
    'effort': {'v1': 'Simple allocation', 'v2': 'Manager with justification', 'v3': 'Strict, with examples and range'},
    'ask_anything': {'v1': 'Crisp, direct answer', 'v2': 'Friendly, detailed answer', 'v3': 'List-style answer'},
}

def get_prompt(module, version=None):
    if module == 'email':
        return EMAIL_REWRITE_PROMPTS[version]
    elif module == 'markdown':
        return MARKDOWN_GEN_PROMPTS[version]
    elif module == 'qa_eval':
        return QA_EVAL_PROMPTS[version]
    elif module == 'qa_gen':
        return QA_GEN_PROMPTS[version]
    elif module == 'effort':
        return EFFORT_ESTIMATE_PROMPTS[version]
    elif module == 'ask_anything':
        return ASK_ANYTHING_PROMPTS[version]
    else:
        raise ValueError(f"Unknown module: {module}") 