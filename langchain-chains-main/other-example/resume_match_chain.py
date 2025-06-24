from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Branch 1: Check skill match
prompt_skill = PromptTemplate(
    template='Given the following resume and job description, list the key skills from the resume that match the job requirements.\nResume:\n{resume}\nJob Description:\n{job_description}',
    input_variables=['resume', 'job_description']
)

# Branch 2: Evaluate experience level
prompt_experience = PromptTemplate(
    template='Based on the following resume, evaluate the candidate\'s experience level (e.g., Junior, Mid, Senior):\n{resume}',
    input_variables=['resume']
)

# Parallel chain for skill match and experience evaluation
parallel_chain = RunnableParallel({
    'skill_match': prompt_skill | model | parser,
    'experience_level': prompt_experience | model | parser
})

# Combine and generate recommendation summary
prompt_recommend = PromptTemplate(
    template='Given the skill match and experience level below, write a recommendation summary for the candidate (e.g., "Strong fit", "Needs more experience", etc.):\nSkill Match: {skill_match}\nExperience Level: {experience_level}',
    input_variables=['skill_match', 'experience_level']
)

recommend_chain = prompt_recommend | model | parser

# Full chain: parallel extraction -> recommendation
chain = parallel_chain | recommend_chain

resume = '''\
John Doe
Software Engineer with 5 years of experience in Python, machine learning, and cloud computing. Led several projects using AWS and Docker. Skilled in data analysis, REST APIs, and agile methodologies.
'''

job_description = '''\
Looking for a Senior Software Engineer with strong experience in Python, cloud platforms (preferably AWS), and machine learning. Must have experience with Docker and leading technical projects.
'''

result = chain.invoke({'resume': resume, 'job_description': job_description})

print(result)

chain.get_graph().print_ascii() 