from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# RootCauseAnalysis schema
class RootCauseAnalysis(BaseModel):
    incident_title: str = Field(description="Title or brief description of the incident")
    date: Optional[str] = Field(default=None, description="Date of the incident")
    affected_systems: Optional[list[str]] = Field(default=None, description="List of affected systems/components")
    summary: str = Field(description="Brief summary of what happened")
    root_cause: str = Field(description="The identified root cause")
    contributing_factors: Optional[list[str]] = Field(default=None, description="List of contributing factors")
    corrective_actions: list[str] = Field(description="Actions taken or recommended to prevent recurrence")
    impact: Optional[str] = Field(default=None, description="Description of the impact")
    reported_by: Optional[str] = Field(default=None, description="Name of the person reporting")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Overall sentiment: use only 'pos', 'neg', or 'neutral'")

structured_model = model.with_structured_output(RootCauseAnalysis)

# Example root cause analysis
rca_report = """
Yesterday at 2 PM, our main website was unavailable for nearly an hour. Users couldn't log in or make purchases. The problem was caused by a backup script that locked database tables. We didn't have monitoring for this, so it took a while to notice. The authentication and order systems were both affected. We fixed the script and added alerts for future issues. This outage led to lost sales and customer complaints. Maria Lopez from DevOps investigated. Overall, it was a negative experience.
"""

result = structured_model.invoke(rca_report)
print(result) 