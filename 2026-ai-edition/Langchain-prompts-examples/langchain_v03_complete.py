# 14_complete_v0.3_production_app.py
# Complete production-ready application using LangChain 0.3+ latest patterns
# This demonstrates best practices with the newest LangChain features

"""
SETUP:
pip install langchain==0.3.0 langchain-openai==0.2.0 langchain-anthropic==0.3.0
pip install langchain-core==0.3.0 langchain-community==0.3.0
pip install python-dotenv pydantic

Create .env:
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Literal
from datetime import datetime

# Modern LangChain 0.3+ imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel, 
    RunnableLambda,
    RunnableSerializable
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field, validator

load_dotenv()

# ==================== PRODUCTION APP: AI-POWERED CONTENT STUDIO ====================
print("="*70)
print("PRODUCTION APPLICATION: AI CONTENT STUDIO")
print("Built with LangChain 0.3+ Modern Patterns")
print("="*70 + "\n")

# ==================== 1. DATA MODELS ====================
class ContentBrief(BaseModel):
    """Input brief for content creation"""
    topic: str = Field(description="Main topic to write about")
    tone: Literal["professional", "casual", "technical", "friendly"] = Field(
        default="professional",
        description="Tone of the content"
    )
    target_audience: str = Field(description="Who is this content for?")
    content_type: Literal["blog", "social", "email", "documentation"] = Field(
        description="Type of content to create"
    )
    keywords: List[str] = Field(default=[], description="SEO keywords to include")
    word_count: int = Field(default=500, description="Target word count")

class ContentOutline(BaseModel):
    """Structured outline for content"""
    title: str = Field(description="Compelling title")
    hook: str = Field(description="Opening hook to grab attention")
    main_points: List[str] = Field(description="Key points to cover")
    conclusion_idea: str = Field(description="How to wrap up")
    estimated_sections: int = Field(description="Number of sections needed")

class GeneratedContent(BaseModel):
    """Final generated content"""
    title: str = Field(description="Final title")
    content: str = Field(description="Full content body")
    meta_description: str = Field(description="SEO meta description")
    tags: List[str] = Field(description="Relevant tags")
    word_count: int = Field(description="Actual word count")
    readability_score: str = Field(description="Reading level estimate")

class ContentFeedback(BaseModel):
    """Quality assessment of content"""
    quality_score: int = Field(ge=1, le=10, description="Overall quality (1-10)")
    strengths: List[str] = Field(description="What works well")
    improvements: List[str] = Field(description="What could be better")
    seo_score: int = Field(ge=1, le=10, description="SEO optimization score")
    recommendations: List[str] = Field(description="Specific improvement suggestions")

# ==================== 2. CONTENT STUDIO CLASS ====================
class AIContentStudio:
    """
    Production-ready content generation system using LangChain 0.3+
    Features: Multi-step generation, quality control, revision support
    """
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        """Initialize the content studio"""
        
        # Initialize LLM with configuration
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=3,
            request_timeout=60
        )
        
        # Alternative: Use Claude
        # self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=temperature)
        
        # Initialize conversation memory store
        self.conversation_store: Dict[str, ChatMessageHistory] = {}
        
        # Build all chains
        self._build_chains()
    
    def _build_chains(self):
        """Build all LCEL chains for the studio"""
        
        # CHAIN 1: Outline Generation
        outline_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content strategist.
Create a detailed outline for content based on the brief.
Consider the target audience, tone, and content type."""),
            ("human", """Create an outline for:

Topic: {topic}
Tone: {tone}
Audience: {target_audience}
Type: {content_type}
Keywords: {keywords}
Target Length: {word_count} words

{format_instructions}""")
        ])
        
        outline_parser = PydanticOutputParser(pydantic_object=ContentOutline)
        
        self.outline_chain = (
            {
                "topic": lambda x: x["topic"],
                "tone": lambda x: x["tone"],
                "target_audience": lambda x: x["target_audience"],
                "content_type": lambda x: x["content_type"],
                "keywords": lambda x: ", ".join(x["keywords"]) if x["keywords"] else "None",
                "word_count": lambda x: x["word_count"],
                "format_instructions": lambda x: outline_parser.get_format_instructions()
            }
            | outline_prompt
            | self.llm
            | outline_parser
        )
        
        # CHAIN 2: Content Generation
        content_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional content writer.
Write engaging, high-quality content based on the outline.
Match the specified tone and target the right audience.
Include keywords naturally."""),
            ("human", """Write content based on this outline:

Title: {title}
Hook: {hook}
Main Points: {main_points}
Conclusion: {conclusion_idea}

Parameters:
- Tone: {tone}
- Audience: {target_audience}
- Keywords to include: {keywords}
- Target word count: {word_count}

{format_instructions}""")
        ])
        
        content_parser = PydanticOutputParser(pydantic_object=GeneratedContent)
        
        self.content_chain = (
            {
                "title": lambda x: x["outline"].title,
                "hook": lambda x: x["outline"].hook,
                "main_points": lambda x: "\n".join(f"- {p}" for p in x["outline"].main_points),
                "conclusion_idea": lambda x: x["outline"].conclusion_idea,
                "tone": lambda x: x["brief"].tone,
                "target_audience": lambda x: x["brief"].target_audience,
                "keywords": lambda x: ", ".join(x["brief"].keywords) if x["brief"].keywords else "None",
                "word_count": lambda x: x["brief"].word_count,
                "format_instructions": lambda x: content_parser.get_format_instructions()
            }
            | content_prompt
            | self.llm
            | content_parser
        )
        
        # CHAIN 3: Quality Review
        review_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content quality reviewer and SEO expert.
Evaluate content thoroughly and provide actionable feedback."""),
            ("human", """Review this content:

TITLE: {title}
CONTENT:
{content}

TARGET KEYWORDS: {keywords}
TARGET AUDIENCE: {target_audience}

{format_instructions}""")
        ])
        
        review_parser = PydanticOutputParser(pydantic_object=ContentFeedback)
        
        self.review_chain = (
            {
                "title": lambda x: x["title"],
                "content": lambda x: x["content"],
                "keywords": lambda x: ", ".join(x.get("keywords", [])),
                "target_audience": lambda x: x.get("target_audience", "general"),
                "format_instructions": lambda x: review_parser.get_format_instructions()
            }
            | review_prompt
            | self.llm
            | review_parser
        )
        
        # CHAIN 4: Revision Chain
        revision_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content editor. Improve content based on feedback."),
            ("human", """Original Content:
{original_content}

Feedback:
{feedback}

Revise the content addressing the feedback while maintaining quality.
Return only the revised content.""")
        ])
        
        self.revision_chain = (
            revision_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_content(self, brief: ContentBrief, review: bool = True) -> Dict:
        """
        Main method to generate content
        
        Args:
            brief: Content brief with requirements
            review: Whether to include quality review
            
        Returns:
            Dictionary with content and optional review
        """
        print(f"\n📝 Generating content about: {brief.topic}")
        print(f"   Tone: {brief.tone} | Audience: {brief.target_audience}")
        print("-"*70)
        
        # Step 1: Generate outline
        print("\n1️⃣  Creating outline...")
        outline = self.outline_chain.invoke(brief.model_dump())
        print(f"   ✓ Outline created with {outline.estimated_sections} sections")
        
        # Step 2: Generate content
        print("\n2️⃣  Writing content...")
        content = self.content_chain.invoke({
            "brief": brief,
            "outline": outline
        })
        print(f"   ✓ Content generated ({content.word_count} words)")
        
        result = {
            "outline": outline,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 3: Optional review
        if review:
            print("\n3️⃣  Reviewing quality...")
            feedback = self.review_chain.invoke({
                "title": content.title,
                "content": content.content,
                "keywords": brief.keywords,
                "target_audience": brief.target_audience
            })
            result["feedback"] = feedback
            print(f"   ✓ Quality Score: {feedback.quality_score}/10")
            print(f"   ✓ SEO Score: {feedback.seo_score}/10")
        
        print("\n" + "="*70)
        return result
    
    def revise_content(self, original_content: str, feedback: ContentFeedback) -> str:
        """Revise content based on feedback"""
        print("\n🔄 Revising content based on feedback...")
        
        feedback_text = "\n".join([
            "Improvements needed:",
            *[f"- {imp}" for imp in feedback.improvements],
            "\nRecommendations:",
            *[f"- {rec}" for rec in feedback.recommendations]
        ])
        
        revised = self.revision_chain.invoke({
            "original_content": original_content,
            "feedback": feedback_text
        })
        
        print("   ✓ Content revised")
        return revised
    
    async def generate_content_async(self, brief: ContentBrief) -> Dict:
        """Async version of content generation"""
        print(f"\n⚡ Async generating content about: {brief.topic}")
        
        # All chains support async
        outline = await self.outline_chain.ainvoke(brief.model_dump())
        content = await self.content_chain.ainvoke({
            "brief": brief,
            "outline": outline
        })
        
        print(f"   ✓ Async generation complete ({content.word_count} words)")
        
        return {"outline": outline, "content": content}
    
    def batch_generate(self, briefs: List[ContentBrief]) -> List[Dict]:
        """Generate multiple content pieces efficiently"""
        print(f"\n📦 Batch generating {len(briefs)} pieces of content...")
        
        # Parallel outline generation
        outlines = self.outline_chain.batch([b.model_dump() for b in briefs])
        
        # Parallel content generation
        contents = self.content_chain.batch([
            {"brief": brief, "outline": outline}
            for brief, outline in zip(briefs, outlines)
        ])
        
        print(f"   ✓ Batch generation complete")
        
        return [
            {"outline": outline, "content": content}
            for outline, content in zip(outlines, contents)
        ]
    
    def stream_content_generation(self, brief: ContentBrief):
        """Stream content generation in real-time"""
        print(f"\n🔴 LIVE: Streaming content generation...")
        print("-"*70)
        
        # First get outline (not streamed)
        outline = self.outline_chain.invoke(brief.model_dump())
        
        # Stream content generation
        for chunk in self.content_chain.stream({
            "brief": brief,
            "outline": outline
        }):
            # Stream the content as it's generated
            if hasattr(chunk, 'content'):
                print(chunk.content, end="", flush=True)
        
        print("\n" + "-"*70)

# ==================== 3. EXAMPLE USAGE ====================
def main():
    """Run the content studio demo"""
    
    # Initialize studio
    studio = AIContentStudio(model="gpt-4", temperature=0.7)
    
    # Example 1: Blog Post
    print("\n" + "="*70)
    print("EXAMPLE 1: BLOG POST GENERATION")
    print("="*70)
    
    blog_brief = ContentBrief(
        topic="The Impact of AI on Software Development",
        tone="professional",
        target_audience="software developers and tech leaders",
        content_type="blog",
        keywords=["AI", "software development", "automation", "productivity"],
        word_count=800
    )
    
    result = studio.generate_content(blog_brief, review=True)
    
    # Display results
    print("\n📄 GENERATED CONTENT:")
    print("="*70)
    print(f"Title: {result['content'].title}")
    print(f"Meta: {result['content'].meta_description}")
    print(f"Tags: {', '.join(result['content'].tags)}")
    print(f"\nContent Preview:")
    print(result['content'].content[:300] + "...")
    
    if 'feedback' in result:
        print(f"\n📊 QUALITY REVIEW:")
        print("="*70)
        print(f"Overall Quality: {result['feedback'].quality_score}/10")
        print(f"SEO Score: {result['feedback'].seo_score}/10")
        print(f"\nStrengths:")
        for strength in result['feedback'].strengths:
            print(f"  ✓ {strength}")
        print(f"\nImprovements:")
        for imp in result['feedback'].improvements:
            print(f"  • {imp}")
    
    # Example 2: Revision
    if result['feedback'].quality_score < 8:
        print("\n" + "="*70)
        print("EXAMPLE 2: CONTENT REVISION")
        print("="*70)
        
        revised = studio.revise_content(
            result['content'].content,
            result['feedback']
        )
        print(f"\nRevised Content Preview:")
        print(revised[:300] + "...")
    
    # Example 3: Batch Generation
    print("\n" + "="*70)
    print("EXAMPLE 3: BATCH GENERATION")
    print("="*70)
    
    briefs = [
        ContentBrief(
            topic="Python Tips for Beginners",
            tone="friendly",
            target_audience="beginner programmers",
            content_type="blog",
            word_count=500
        ),
        ContentBrief(
            topic="Cloud Security Best Practices",
            tone="technical",
            target_audience="DevOps engineers",
            content_type="documentation",
            word_count=600
        )
    ]
    
    batch_results = studio.batch_generate(briefs)
    
    for i, result in enumerate(batch_results, 1):
        print(f"\n{i}. {result['content'].title}")
        print(f"   Words: {result['content'].word_count}")

# ==================== 4. ASYNC EXAMPLE ====================
async def async_demo():
    """Demonstrate async content generation"""
    import asyncio
    
    studio = AIContentStudio(model="gpt-3.5-turbo")
    
    brief = ContentBrief(
        topic="Getting Started with Docker",
        tone="technical",
        target_audience="developers",
        content_type="documentation",
        word_count=500
    )
    
    result = await studio.generate_content_async(brief)
    print(f"\nAsync Result: {result['content'].title}")

# Run async example
def run_async_demo():
    import asyncio
    asyncio.run(async_demo())

# ==================== RUN THE APPLICATION ====================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              AI CONTENT STUDIO - PRODUCTION READY                    ║
║              Built with LangChain 0.3+ (Latest)                      ║
╚══════════════════════════════════════════════════════════════════════╝

Features:
✓ Multi-step content generation (outline → content → review)
✓ Quality assessment and revision support
✓ Batch processing for efficiency
✓ Async operations for scalability
✓ Streaming for real-time output
✓ Type-safe with Pydantic models
✓ Production-ready error handling

Run the demo:
    python 14_complete_v0.3_production_app.py
    """)
    
    # Uncomment to run:
    # main()
    # run_async_demo()