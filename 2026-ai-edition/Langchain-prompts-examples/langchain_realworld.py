# 09_real_world_examples.py
# Real-world applications combining multiple LangChain prompt techniques
# These are production-ready patterns you'd use in actual projects

from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from datetime import datetime
from typing import List, Dict

# ==================== EXAMPLE 1: CUSTOMER SUPPORT CHATBOT ====================
print("="*60)
print("EXAMPLE 1: CUSTOMER SUPPORT CHATBOT")
print("="*60 + "\n")

# Multi-tier support system with escalation
class CustomerSupportPromptSystem:
    """Complete customer support prompt system"""
    
    def __init__(self):
        # Tier 1: Basic support
        self.tier1_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Tier 1 customer support agent.
- Be friendly and empathetic
- Handle common issues: password resets, billing questions, account info
- If issue is complex, recommend escalation to Tier 2
- Always maintain professional tone

Current date: {date}
Customer tier: {customer_tier}"""),
            ("human", "Customer message: {message}")
        ])
        
        # Tier 2: Technical support
        self.tier2_template = ChatPromptTemplate.from_messages([
            ("system", """You are a Tier 2 technical support specialist.
- Handle technical issues and bugs
- Request logs and diagnostic information when needed
- Provide step-by-step troubleshooting
- Escalate to engineering if it's a system-level issue

Previous conversation: {conversation_history}
Issue category: {issue_category}"""),
            ("human", "Technical issue: {message}")
        ])
    
    def get_prompt(self, tier: int, **kwargs) -> str:
        """Get appropriate prompt based on support tier"""
        # Add current date
        kwargs['date'] = datetime.now().strftime("%Y-%m-%d")
        
        if tier == 1:
            return self.tier1_template.format_messages(**kwargs)
        elif tier == 2:
            return self.tier2_template.format_messages(**kwargs)

# Example usage
support_system = CustomerSupportPromptSystem()

# Tier 1 interaction
tier1_prompt = support_system.get_prompt(
    tier=1,
    customer_tier="Premium",
    message="I forgot my password and can't log in"
)
print("Tier 1 Support Prompt:")
for msg in tier1_prompt:
    print(f"{msg.type.upper()}: {msg.content}\n")

# Tier 2 interaction
tier2_prompt = support_system.get_prompt(
    tier=2,
    conversation_history="User reported login issues. Tier 1 verified password is correct.",
    issue_category="Authentication",
    message="Getting 'Session expired' error immediately after login"
)
print("Tier 2 Support Prompt:")
for msg in tier2_prompt:
    print(f"{msg.type.upper()}: {msg.content}\n")

# ==================== EXAMPLE 2: CONTENT GENERATION PIPELINE ====================
print("\n" + "="*60)
print("EXAMPLE 2: BLOG POST GENERATION PIPELINE")
print("="*60 + "\n")

# Multi-stage content creation
class BlogPostGenerator:
    """Generate blog posts through multiple stages"""
    
    def __init__(self):
        # Stage 1: Research and outline
        self.research_template = PromptTemplate.from_template("""
You are a content researcher.

TOPIC: {topic}
TARGET AUDIENCE: {audience}
KEYWORDS: {keywords}

Create a detailed outline for a blog post including:
1. Compelling title options (3)
2. Main sections with subpoints
3. Key points to cover
4. Research questions to address
""")
        
        # Stage 2: Writing
        self.writing_template = PromptTemplate.from_template("""
You are a professional content writer.

OUTLINE:
{outline}

TONE: {tone}
WORD COUNT: {word_count}

Write the complete blog post following the outline.
- Use engaging introductions and conclusions
- Include transitions between sections
- Add relevant examples
- Maintain consistent tone
""")
        
        # Stage 3: SEO optimization
        self.seo_template = PromptTemplate.from_template("""
You are an SEO specialist.

BLOG POST:
{blog_post}

TARGET KEYWORDS: {keywords}

Optimize this blog post for SEO:
1. Add meta description (150-160 chars)
2. Suggest title tags
3. Recommend header structure (H1, H2, H3)
4. Identify keyword placement opportunities
5. Suggest internal linking ideas
""")
    
    def generate_outline(self, topic: str, audience: str, keywords: List[str]) -> str:
        """Stage 1: Generate outline"""
        return self.research_template.format(
            topic=topic,
            audience=audience,
            keywords=", ".join(keywords)
        )
    
    def generate_content(self, outline: str, tone: str, word_count: int) -> str:
        """Stage 2: Generate content"""
        return self.writing_template.format(
            outline=outline,
            tone=tone,
            word_count=word_count
        )
    
    def optimize_seo(self, blog_post: str, keywords: List[str]) -> str:
        """Stage 3: SEO optimization"""
        return self.seo_template.format(
            blog_post=blog_post,
            keywords=", ".join(keywords)
        )

# Example usage
blog_gen = BlogPostGenerator()

outline_prompt = blog_gen.generate_outline(
    topic="The Future of Remote Work",
    audience="HR professionals and business leaders",
    keywords=["remote work", "hybrid teams", "productivity", "work-life balance"]
)
print("Stage 1 - Research & Outline Prompt:")
print(outline_prompt)
print("\n" + "-"*60 + "\n")

# Simulated outline from Stage 1
simulated_outline = """
Title Options:
1. "The Future of Remote Work: Trends Shaping 2025"
2. "Remote Work Revolution: What HR Leaders Need to Know"
3. "Hybrid Teams: The New Normal for Modern Business"

Main Sections:
I. Introduction - The shift to remote work
II. Current trends in remote work
III. Challenges and solutions
IV. Future predictions
V. Conclusion - Action steps
"""

content_prompt = blog_gen.generate_content(
    outline=simulated_outline,
    tone="professional but conversational",
    word_count=1500
)
print("Stage 2 - Content Writing Prompt:")
print(content_prompt)

# ==================== EXAMPLE 3: DATA ANALYSIS ASSISTANT ====================
print("\n" + "="*60)
print("EXAMPLE 3: DATA ANALYSIS ASSISTANT")
print("="*60 + "\n")

class DataAnalysisPromptSystem:
    """Prompts for data analysis tasks"""
    
    def __init__(self):
        # Few-shot examples for data analysis
        self.analysis_examples = [
            {
                "data": "Sales Q1: $100K, Q2: $150K, Q3: $140K, Q4: $180K",
                "analysis": """
Key Findings:
- 80% annual growth from Q1 to Q4
- Strong momentum with Q2 (+50%) and Q4 (+28%)
- Minor dip in Q3 (-6.7%) - investigate seasonal factors
- Overall positive trend indicating healthy business growth

Recommendations:
- Investigate Q3 dip to prevent future occurrences
- Capitalize on Q4 success factors for Q1 planning
- Set Q1 target at $195K (8% growth based on trend)
"""
            },
            {
                "data": "User engagement: Mon-25%, Tue-30%, Wed-28%, Thu-35%, Fri-22%, Sat-15%, Sun-18%",
                "analysis": """
Key Findings:
- Peak engagement Thursday (35%)
- Weekday average: 28% vs Weekend average: 16.5%
- Friday shows unexpected drop to 22%
- Clear weekday preference pattern

Recommendations:
- Schedule major campaigns for Tuesday-Thursday
- Investigate Friday drop - possible fatigue factor
- Develop weekend-specific content strategy
- Consider Monday re-engagement campaigns
"""
            }
        ]
        
        example_formatter = PromptTemplate(
            input_variables=["data", "analysis"],
            template="DATA: {data}\n\nANALYSIS:\n{analysis}"
        )
        
        self.analysis_prompt = FewShotPromptTemplate(
            examples=self.analysis_examples,
            example_prompt=example_formatter,
            prefix="""You are a data analyst. Provide clear, actionable insights.

For each dataset:
1. Identify key findings and trends
2. Calculate relevant metrics
3. Provide specific, actionable recommendations

Examples:""",
            suffix="\nDATA: {data}\n\nANALYSIS:",
            input_variables=["data"]
        )
    
    def analyze(self, data: str) -> str:
        """Generate analysis prompt"""
        return self.analysis_prompt.format(data=data)

# Example usage
data_analyzer = DataAnalysisPromptSystem()

analysis_prompt = data_analyzer.analyze(
    "Website traffic: Jan-45K, Feb-52K, Mar-48K, Apr-61K, May-58K, Jun-70K"
)
print("Data Analysis Prompt with Few-Shot Examples:")
print(analysis_prompt)

# ==================== EXAMPLE 4: CODE REVIEW SYSTEM ====================
print("\n" + "="*60)
print("EXAMPLE 4: CODE REVIEW SYSTEM")
print("="*60 + "\n")

class CodeReviewPromptSystem:
    """Comprehensive code review prompts"""
    
    def __init__(self):
        # Base review template
        review_base = PromptTemplate.from_template("""
You are an expert code reviewer specializing in {language}.
""")
        
        # Security focus
        security_focus = PromptTemplate.from_template("""
SECURITY REVIEW FOCUS:
- Check for SQL injection vulnerabilities
- Verify input validation and sanitization
- Review authentication and authorization
- Check for sensitive data exposure
- Identify potential XSS vulnerabilities
""")
        
        # Performance focus
        performance_focus = PromptTemplate.from_template("""
PERFORMANCE REVIEW FOCUS:
- Identify inefficient algorithms (O(n²) or worse)
- Check for unnecessary loops or computations
- Review database query efficiency
- Look for memory leaks
- Identify blocking operations
""")
        
        # Code quality focus
        quality_focus = PromptTemplate.from_template("""
CODE QUALITY FOCUS:
- Assess code readability and maintainability
- Check naming conventions
- Review error handling
- Evaluate test coverage
- Identify code duplication
""")
        
        # The code to review
        code_template = PromptTemplate.from_template("""

CODE TO REVIEW:
```{language}
{code}
```

Provide detailed feedback with:
1. Critical issues (must fix)
2. Suggestions (should fix)
3. Optional improvements
4. Positive aspects

Format each point with line numbers and specific examples.
""")
        
        # Final combined template
        final_template = PromptTemplate.from_template("""
{base}
{focus_areas}
{code_section}
""")
        
        # Create pipeline
        self.review_pipeline = PipelinePromptTemplate(
            final_prompt=final_template,
            pipeline_prompts=[
                ("base", review_base),
                ("code_section", code_template),
            ]
        )
        
        self.security_pipeline = PipelinePromptTemplate(
            final_prompt=final_template,
            pipeline_prompts=[
                ("base", review_base),
                ("focus_areas", security_focus),
                ("code_section", code_template),
            ]
        )
        
        self.performance_pipeline = PipelinePromptTemplate(
            final_prompt=final_template,
            pipeline_prompts=[
                ("base", review_base),
                ("focus_areas", performance_focus),
                ("code_section", code_template),
            ]
        )
    
    def review_code(self, code: str, language: str, focus: str = "general") -> str:
        """Generate code review prompt with specific focus"""
        params = {"code": code, "language": language}
        
        if focus == "security":
            return self.security_pipeline.format(**params)
        elif focus == "performance":
            return self.performance_pipeline.format(**params)
        else:
            return self.review_pipeline.format(**params)

# Example usage
code_reviewer = CodeReviewPromptSystem()

sample_code = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query)
    return result
"""

security_review = code_reviewer.review_code(
    code=sample_code,
    language="python",
    focus="security"
)
print("Security-Focused Code Review Prompt:")
print(security_review)

# ==================== EXAMPLE 5: PERSONALIZED LEARNING SYSTEM ====================
print("\n" + "="*60)
print("EXAMPLE 5: PERSONALIZED LEARNING SYSTEM")
print("="*60 + "\n")

class PersonalizedLearningSystem:
    """Adaptive learning prompts based on user progress"""
    
    def __init__(self):
        self.difficulty_levels = {
            "beginner": {
                "explanation_style": "simple with lots of examples",
                "code_complexity": "basic",
                "concepts": "one at a time"
            },
            "intermediate": {
                "explanation_style": "balanced with practical examples",
                "code_complexity": "moderate with some advanced patterns",
                "concepts": "multiple related concepts"
            },
            "advanced": {
                "explanation_style": "concise with edge cases",
                "code_complexity": "advanced patterns and optimizations",
                "concepts": "complex interconnected concepts"
            }
        }
    
    def create_lesson_prompt(
        self,
        topic: str,
        user_level: str,
        learning_style: str,
        previous_mistakes: List[str] = None
    ) -> str:
        """Create personalized lesson prompt"""
        
        level_config = self.difficulty_levels[user_level]
        
        prompt_parts = [
            f"You are a programming tutor teaching {topic}.",
            f"\nSTUDENT LEVEL: {user_level}",
            f"EXPLANATION STYLE: {level_config['explanation_style']}",
            f"CODE COMPLEXITY: {level_config['code_complexity']}",
            f"FOCUS: {level_config['concepts']}",
            f"\nLEARNING STYLE: {learning_style}",
        ]
        
        if previous_mistakes:
            prompt_parts.append("\nCOMMON MISTAKES TO ADDRESS:")
            for mistake in previous_mistakes:
                prompt_parts.append(f"- {mistake}")
        
        prompt_parts.append(f"\nExplain {topic} with appropriate examples and exercises.")
        
        return "\n".join(prompt_parts)

# Example usage
learning_system = PersonalizedLearningSystem()

beginner_prompt = learning_system.create_lesson_prompt(
    topic="Python list comprehensions",
    user_level="beginner",
    learning_style="visual with diagrams",
    previous_mistakes=[
        "Confuses list comprehension syntax with regular loops",
        "Forgets to include the loop variable"
    ]
)

print("Personalized Learning Prompt (Beginner):")
print(beginner_prompt)
print("\n" + "-"*60 + "\n")

advanced_prompt = learning_system.create_lesson_prompt(
    topic="Python decorators",
    user_level="advanced",
    learning_style="hands-on with real-world scenarios"
)

print("Personalized Learning Prompt (Advanced):")
print(advanced_prompt)