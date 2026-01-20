# 12_practical_llm_applications.py
# Complete working applications using LLMs with prompts
# These are production-ready examples you can adapt for real projects

"""
SETUP:
pip install langchain langchain-openai langchain-anthropic python-dotenv

Create .env file:
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
"""

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# ==================== APPLICATION 1: EMAIL ASSISTANT ====================
print("="*70)
print("APPLICATION 1: INTELLIGENT EMAIL ASSISTANT")
print("="*70 + "\n")

class EmailAssistant:
    """Generate professional emails based on context"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        self.email_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional email writing assistant.
Write clear, concise, and appropriately toned emails.
Include:
- Proper greeting
- Clear message body
- Professional closing
- Signature line placeholder"""),
            ("human", """Write an email with these details:
To: {recipient}
Purpose: {purpose}
Tone: {tone}
Key points to include:
{key_points}""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.email_prompt)
    
    def generate_email(self, recipient, purpose, tone, key_points):
        """Generate a professional email"""
        result = self.chain.run(
            recipient=recipient,
            purpose=purpose,
            tone=tone,
            key_points=key_points
        )
        return result

# Example usage
def demo_email_assistant():
    assistant = EmailAssistant()
    
    email = assistant.generate_email(
        recipient="Project Team",
        purpose="Request deadline extension",
        tone="professional and respectful",
        key_points="""
        - Original deadline: Feb 15
        - Requesting extension to Feb 28
        - Reason: Additional testing requirements discovered
        - Commitment to quality delivery
        """
    )
    
    print("GENERATED EMAIL:")
    print("-"*70)
    print(email)
    print("-"*70 + "\n")

# Uncomment to run:
# demo_email_assistant()

# ==================== APPLICATION 2: CODE REVIEWER ====================
print("APPLICATION 2: AUTOMATED CODE REVIEWER")
print("="*70 + "\n")

class CodeReview(BaseModel):
    """Structured code review output"""
    overall_quality: str = Field(description="Overall code quality rating (Poor/Fair/Good/Excellent)")
    critical_issues: List[str] = Field(description="Critical issues that must be fixed")
    suggestions: List[str] = Field(description="Improvement suggestions")
    positive_aspects: List[str] = Field(description="Good practices found")
    security_concerns: List[str] = Field(description="Security issues if any")

class CodeReviewer:
    """Automated code review system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.parser = PydanticOutputParser(pydantic_object=CodeReview)
        
        self.review_prompt = PromptTemplate(
            template="""You are an expert code reviewer.

Review this {language} code:

```{language}
{code}
```

Focus on:
- Code quality and readability
- Potential bugs
- Security vulnerabilities
- Performance issues
- Best practices

{format_instructions}""",
            input_variables=["language", "code"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.review_prompt,
            output_parser=self.parser
        )
    
    def review_code(self, code, language="python"):
        """Review code and return structured feedback"""
        result = self.chain.run(code=code, language=language)
        return result

# Example usage
def demo_code_reviewer():
    reviewer = CodeReviewer()
    
    sample_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result

def process_payment(amount):
    if amount > 0:
        charge_card(amount)
        return True
    """
    
    review = reviewer.review_code(sample_code)
    
    print("CODE REVIEW RESULTS:")
    print("-"*70)
    print(f"Overall Quality: {review.overall_quality}")
    print(f"\nCritical Issues ({len(review.critical_issues)}):")
    for issue in review.critical_issues:
        print(f"  - {issue}")
    print(f"\nSuggestions ({len(review.suggestions)}):")
    for suggestion in review.suggestions:
        print(f"  - {suggestion}")
    print(f"\nSecurity Concerns ({len(review.security_concerns)}):")
    for concern in review.security_concerns:
        print(f"  - {concern}")
    print("-"*70 + "\n")

# Uncomment to run:
# demo_code_reviewer()

# ==================== APPLICATION 3: CUSTOMER SUPPORT BOT ====================
print("APPLICATION 3: CUSTOMER SUPPORT CHATBOT")
print("="*70 + "\n")

from langchain.memory import ConversationBufferMemory

class SupportBot:
    """Customer support chatbot with memory"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.memory = ConversationBufferMemory(return_messages=True)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer support agent for TechCorp.

Guidelines:
- Be friendly and professional
- Ask clarifying questions if needed
- Provide step-by-step solutions
- Escalate to human agent if issue is complex
- Always thank the customer

Common issues you can help with:
- Password resets
- Account billing
- Product features
- Technical troubleshooting

Company info:
- Support hours: 9 AM - 6 PM EST
- Email: support@techcorp.com
- Phone: 1-800-TECH-HELP"""),
            ("human", "{input}")
        ])
        
        from langchain.chains import ConversationChain
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
    
    def chat(self, user_message):
        """Process user message and return response"""
        response = self.conversation.predict(input=user_message)
        return response
    
    def reset_conversation(self):
        """Start a new conversation"""
        self.memory.clear()

# Example usage
def demo_support_bot():
    bot = SupportBot()
    
    conversation = [
        "Hi, I can't log into my account",
        "I tried resetting my password but didn't get the email",
        "It's john.doe@email.com",
        "Thanks for your help!"
    ]
    
    print("CUSTOMER SUPPORT CONVERSATION:")
    print("-"*70)
    for message in conversation:
        print(f"\nCUSTOMER: {message}")
        response = bot.chat(message)
        print(f"SUPPORT: {response}")
        print("-"*70)

# Uncomment to run:
# demo_support_bot()

# ==================== APPLICATION 4: CONTENT SUMMARIZER ====================
print("APPLICATION 4: INTELLIGENT CONTENT SUMMARIZER")
print("="*70 + "\n")

class Summary(BaseModel):
    """Structured summary output"""
    main_points: List[str] = Field(description="Key points from the content")
    summary: str = Field(description="Concise summary in 2-3 sentences")
    keywords: List[str] = Field(description="Important keywords")
    sentiment: str = Field(description="Overall sentiment (Positive/Neutral/Negative)")

class ContentSummarizer:
    """Summarize articles, documents, and content"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.parser = PydanticOutputParser(pydantic_object=Summary)
        
        self.summary_prompt = PromptTemplate(
            template="""Analyze and summarize the following content:

{content}

{format_instructions}""",
            input_variables=["content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt,
            output_parser=self.parser
        )
    
    def summarize(self, content):
        """Generate structured summary"""
        result = self.chain.run(content=content)
        return result

# Example usage
def demo_content_summarizer():
    summarizer = ContentSummarizer()
    
    article = """
    Artificial Intelligence is revolutionizing the healthcare industry in unprecedented ways.
    From diagnosis to treatment, AI-powered systems are helping doctors make more accurate
    decisions. Machine learning algorithms can analyze medical images faster than human
    radiologists, often with higher accuracy rates. Natural language processing is being
    used to extract insights from patient records and research papers. However, challenges
    remain, including data privacy concerns, the need for regulatory frameworks, and
    ensuring AI systems are unbiased. Despite these challenges, the potential benefits
    of AI in healthcare are enormous, promising better patient outcomes and more efficient
    healthcare delivery.
    """
    
    summary = summarizer.summarize(article)
    
    print("SUMMARY RESULTS:")
    print("-"*70)
    print(f"Summary: {summary.summary}")
    print(f"\nMain Points:")
    for point in summary.main_points:
        print(f"  • {point}")
    print(f"\nKeywords: {', '.join(summary.keywords)}")
    print(f"Sentiment: {summary.sentiment}")
    print("-"*70 + "\n")

# Uncomment to run:
# demo_content_summarizer()

# ==================== APPLICATION 5: DATA ANALYZER ====================
print("APPLICATION 5: DATA ANALYSIS ASSISTANT")
print("="*70 + "\n")

class DataAnalysis(BaseModel):
    """Structured data analysis output"""
    trends: List[str] = Field(description="Identified trends in the data")
    insights: List[str] = Field(description="Key insights and findings")
    recommendations: List[str] = Field(description="Actionable recommendations")
    metrics: dict = Field(description="Calculated metrics")

class DataAnalyzer:
    """Analyze data and provide insights"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.parser = PydanticOutputParser(pydantic_object=DataAnalysis)
        
        self.analysis_prompt = PromptTemplate(
            template="""You are a data analyst. Analyze this data:

{data}

Context: {context}

Provide:
1. Trends you observe
2. Key insights
3. Actionable recommendations
4. Important metrics (calculate percentages, growth rates, etc.)

{format_instructions}""",
            input_variables=["data", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt,
            output_parser=self.parser
        )
    
    def analyze(self, data, context=""):
        """Analyze data and return insights"""
        result = self.chain.run(data=data, context=context)
        return result

# Example usage
def demo_data_analyzer():
    analyzer = DataAnalyzer()
    
    sales_data = """
    Monthly Sales (2024):
    Jan: $45,000 (150 transactions)
    Feb: $52,000 (165 transactions)
    Mar: $48,000 (160 transactions)
    Apr: $61,000 (185 transactions)
    May: $58,000 (180 transactions)
    Jun: $70,000 (210 transactions)
    
    Product Categories:
    Electronics: 40%
    Clothing: 30%
    Home & Garden: 20%
    Sports: 10%
    
    Customer Retention: 75%
    Average Order Value: $280
    """
    
    analysis = analyzer.analyze(
        data=sales_data,
        context="E-commerce business focused on retail"
    )
    
    print("DATA ANALYSIS RESULTS:")
    print("-"*70)
    print("\nTRENDS:")
    for trend in analysis.trends:
        print(f"  • {trend}")
    
    print("\nINSIGHTS:")
    for insight in analysis.insights:
        print(f"  • {insight}")
    
    print("\nRECOMMENDATIONS:")
    for rec in analysis.recommendations:
        print(f"  • {rec}")
    
    print("\nMETRICS:")
    for key, value in analysis.metrics.items():
        print(f"  • {key}: {value}")
    print("-"*70 + "\n")

# Uncomment to run:
# demo_data_analyzer()

# ==================== APPLICATION 6: LANGUAGE TUTOR ====================
print("APPLICATION 6: LANGUAGE LEARNING TUTOR")
print("="*70 + "\n")

class LanguageTutor:
    """Interactive language learning assistant"""
    
    def __init__(self, target_language="Spanish", user_level="beginner"):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.target_language = target_language
        self.user_level = user_level
        
        self.tutor_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a patient and encouraging {target_language} tutor.

Student level: {user_level}

Your teaching approach:
- Use simple explanations for beginners
- Provide pronunciation tips
- Give cultural context
- Correct mistakes gently
- Encourage practice
- Use examples relevant to daily life

For each lesson:
1. Teach a concept clearly
2. Provide examples
3. Ask practice questions
4. Give feedback"""),
            ("human", "{input}")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.tutor_prompt)
    
    def teach(self, topic):
        """Teach a specific topic"""
        result = self.chain.run(input=f"Teach me about {topic}")
        return result
    
    def practice(self, user_response, context):
        """Provide feedback on practice"""
        result = self.chain.run(
            input=f"Context: {context}\nStudent said: {user_response}\nProvide feedback."
        )
        return result

# Example usage
def demo_language_tutor():
    tutor = LanguageTutor(target_language="Spanish", user_level="beginner")
    
    # Lesson
    lesson = tutor.teach("basic greetings")
    print("LESSON:")
    print("-"*70)
    print(lesson)
    print("-"*70 + "\n")
    
    # Practice feedback
    feedback = tutor.practice(
        user_response="Hola, como estas?",
        context="Student is practicing greetings"
    )
    print("FEEDBACK:")
    print("-"*70)
    print(feedback)
    print("-"*70 + "\n")

# Uncomment to run:
# demo_language_tutor()

# ==================== MAIN RUNNER ====================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           PRACTICAL LLM APPLICATIONS WITH LANGCHAIN                  ║
╚══════════════════════════════════════════════════════════════════════╝

Available Applications:
1. Email Assistant - Generate professional emails
2. Code Reviewer - Automated code review
3. Support Bot - Customer service chatbot
4. Content Summarizer - Summarize articles/documents
5. Data Analyzer - Analyze data and provide insights
6. Language Tutor - Interactive language learning

To run examples:
1. Uncomment the demo function you want to run
2. Make sure you have API keys in .env file
3. Run: python 12_practical_llm_applications.py

Example:
    demo_email_assistant()
    demo_code_reviewer()
    demo_support_bot()
    demo_content_summarizer()
    demo_data_analyzer()
    demo_language_tutor()

Each application demonstrates:
- Real-world use case
- Proper prompt engineering
- Structured output handling
- Production-ready patterns
    """)