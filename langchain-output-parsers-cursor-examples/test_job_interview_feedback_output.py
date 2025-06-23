from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# JobInterviewFeedback schema
class JobInterviewFeedback(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    position: str = Field(description="Position interviewed for")
    strengths: Optional[list[str]] = Field(default=None, description="Strengths observed during the interview")
    weaknesses: Optional[list[str]] = Field(default=None, description="Weaknesses or areas for improvement")
    overall_impression: str = Field(description="Brief summary of the overall impression")
    recommendation: Literal["hire", "no hire", "hold"] = Field(description="Recommendation: hire, no hire, or hold")
    interviewer: Optional[str] = Field(default=None, description="Name of the interviewer")
    date: Optional[str] = Field(default=None, description="Date of the interview")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Overall sentiment: positive, negative, or neutral")

structured_model = model.with_structured_output(JobInterviewFeedback)

# Example interview feedback
feedback ="""I met with Priya Desai on May 10, 2024 for the Data Analyst position. While she came across as polite and enthusiastic, the overall interview raised some concerns regarding her readiness for this role.
During the case study, Priya struggled to structure her approach and had difficulty interpreting the data effectively. Her explanations lacked clarity, and she often seemed unsure about how to proceed. When we discussed technical concepts, particularly SQL and data modeling, her answers were vague and revealed gaps in foundational knowledge.
She also had limited familiarity with commonly used tools in our tech stack, including basic data visualization platforms and big data technologies, which are important for day-to-day work in this role.
While she may have the potential to grow in the future, I don’t believe she is currently equipped to handle the responsibilities of this position. I’d suggest she continue building her technical skills and gain more hands-on experience before being reconsidered for a similar role."""
result = structured_model.invoke(feedback)
print(result) 