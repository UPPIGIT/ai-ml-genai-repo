# 08_advanced_composition.py
# Advanced techniques for composing and combining prompts
# Includes templating, inheritance, and dynamic prompt generation

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from typing import Dict, List, Any
import json

# Example 1: Template Inheritance Pattern
# Create base templates that can be extended
class BasePromptTemplate:
    """Base template with common structure"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.base_structure = {
            "role": "You are a professional {role}.",
            "context": "Context: {context}",
            "task": "Task: {task}",
            "output": "Output format: {output_format}"
        }
    
    def build_prompt(self, **kwargs) -> str:
        """Build prompt from structure"""
        sections = []
        for key, template in self.base_structure.items():
            if key in kwargs or any(var in kwargs for var in self._extract_variables(template)):
                try:
                    sections.append(template.format(**kwargs))
                except KeyError:
                    pass  # Skip if variables not provided
        return "\n\n".join(sections)
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template"""
        import re
        return re.findall(r'\{(\w+)\}', template)

# Specialized templates that inherit base structure
class CodeReviewTemplate(BasePromptTemplate):
    def __init__(self):
        super().__init__("code_review")
        self.base_structure.update({
            "guidelines": "Guidelines:\n{guidelines}",
            "focus_areas": "Focus on: {focus_areas}"
        })

class ContentWritingTemplate(BasePromptTemplate):
    def __init__(self):
        super().__init__("content_writing")
        self.base_structure.update({
            "tone": "Tone: {tone}",
            "audience": "Target audience: {audience}",
            "word_count": "Word count: {word_count}"
        })

# Use inherited templates
code_template = CodeReviewTemplate()
prompt1 = code_template.build_prompt(
    role="senior software engineer",
    context="Python web application code",
    task="Review this code for security vulnerabilities",
    output_format="Markdown with severity levels",
    guidelines="- Check for SQL injection\n- Verify input validation",
    focus_areas="Security and performance"
)

print("Example 1 - Template Inheritance:")
print(prompt1)
print("\n" + "="*50 + "\n")

# Example 2: Dynamic Prompt Builder
# Build prompts based on configurations
class DynamicPromptBuilder:
    """Build prompts dynamically from configuration"""
    
    def __init__(self):
        self.components = {}
    
    def add_component(self, name: str, template: str, required: bool = False):
        """Add a reusable component"""
        self.components[name] = {
            "template": template,
            "required": required
        }
        return self
    
    def build(self, **kwargs) -> str:
        """Build prompt from components"""
        sections = []
        
        for name, component in self.components.items():
            template = component["template"]
            required = component["required"]
            
            # Extract variables needed for this component
            import re
            variables = re.findall(r'\{(\w+)\}', template)
            
            # Check if all required variables are provided
            has_all_vars = all(var in kwargs for var in variables)
            
            if has_all_vars:
                sections.append(template.format(**kwargs))
            elif required:
                raise ValueError(f"Missing required variables for {name}: {variables}")
        
        return "\n\n".join(sections)

# Build a customer service prompt dynamically
builder = DynamicPromptBuilder()
builder.add_component(
    "persona",
    "You are a {persona_type} customer service representative.",
    required=True
).add_component(
    "customer_info",
    "Customer: {customer_name} (ID: {customer_id})"
).add_component(
    "issue",
    "Issue: {issue_description}",
    required=True
).add_component(
    "priority",
    "Priority Level: {priority}"
).add_component(
    "resolution_goal",
    "Goal: {goal}"
)

prompt2 = builder.build(
    persona_type="friendly and helpful",
    customer_name="Alice Brown",
    customer_id="C-12345",
    issue_description="Cannot log into account",
    priority="High",
    goal="Resolve within 24 hours"
)

print("Example 2 - Dynamic Prompt Builder:")
print(prompt2)
print("\n" + "="*50 + "\n")

# Example 3: Prompt Composition with Mixins
# Mix different behaviors into prompts
class AnalyticalMixin:
    """Add analytical thinking to prompts"""
    analytical_section = """
Analytical Approach:
1. Break down the problem into components
2. Analyze each component systematically
3. Synthesize findings into conclusion
"""

class CreativeMixin:
    """Add creative thinking to prompts"""
    creative_section = """
Creative Approach:
1. Think outside the box
2. Generate multiple unique solutions
3. Combine ideas in novel ways
"""

class EmpatheticMixin:
    """Add empathy to prompts"""
    empathetic_section = """
Empathetic Approach:
1. Consider the human impact
2. Acknowledge emotions and concerns
3. Provide supportive guidance
"""

class ComposedPrompt:
    """Compose prompts with mixins"""
    
    def __init__(self, base_prompt: str, *mixins):
        self.base_prompt = base_prompt
        self.mixins = mixins
    
    def generate(self) -> str:
        sections = [self.base_prompt]
        
        for mixin in self.mixins:
            # Add relevant sections from each mixin
            for attr in dir(mixin):
                if attr.endswith('_section'):
                    sections.append(getattr(mixin, attr))
        
        return "\n".join(sections)

# Create a prompt that's both analytical and empathetic
base = "Help a team resolve a conflict about project priorities."
composed_prompt = ComposedPrompt(base, AnalyticalMixin(), EmpatheticMixin())
prompt3 = composed_prompt.generate()

print("Example 3 - Mixin Composition:")
print(prompt3)
print("\n" + "="*50 + "\n")

# Example 4: Conditional Prompt Assembly
# Assemble prompts based on conditions
class ConditionalPromptAssembler:
    """Assemble prompts with conditional logic"""
    
    def __init__(self):
        self.sections = []
    
    def add_section(self, content: str, condition: callable = None):
        """Add section with optional condition"""
        self.sections.append({
            "content": content,
            "condition": condition if condition else lambda ctx: True
        })
        return self
    
    def assemble(self, context: Dict[str, Any]) -> str:
        """Assemble prompt based on context"""
        active_sections = []
        
        for section in self.sections:
            if section["condition"](context):
                # Format the content with context variables
                try:
                    content = section["content"].format(**context)
                    active_sections.append(content)
                except KeyError:
                    # If formatting fails, use as-is
                    active_sections.append(section["content"])
        
        return "\n\n".join(active_sections)

# Build an adaptive prompt
assembler = ConditionalPromptAssembler()

assembler.add_section(
    "You are a {role}."
).add_section(
    "BEGINNER MODE: Explain concepts simply with examples.",
    condition=lambda ctx: ctx.get("user_level") == "beginner"
).add_section(
    "ADVANCED MODE: Provide technical details and edge cases.",
    condition=lambda ctx: ctx.get("user_level") == "advanced"
).add_section(
    "Time constraint: Complete in {time_limit}",
    condition=lambda ctx: "time_limit" in ctx
).add_section(
    "Task: {task}"
)

# Assemble for beginner with time limit
context1 = {
    "role": "programming tutor",
    "user_level": "beginner",
    "time_limit": "5 minutes",
    "task": "Explain what an API is"
}
prompt4a = assembler.assemble(context1)

print("Example 4a - Conditional Assembly (Beginner):")
print(prompt4a)
print()

# Assemble for advanced without time limit
context2 = {
    "role": "programming tutor",
    "user_level": "advanced",
    "task": "Explain what an API is"
}
prompt4b = assembler.assemble(context2)

print("Example 4b - Conditional Assembly (Advanced):")
print(prompt4b)
print("\n" + "="*50 + "\n")

# Example 5: Prompt Templating with JSON Config
# Store and load prompt configurations
class JSONPromptManager:
    """Manage prompts through JSON configurations"""
    
    def __init__(self):
        self.templates = {}
    
    def load_from_json(self, json_string: str):
        """Load prompt templates from JSON"""
        config = json.loads(json_string)
        self.templates = config.get("templates", {})
    
    def get_prompt(self, template_name: str, **variables) -> str:
        """Get a formatted prompt by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template_config = self.templates[template_name]
        
        # Build prompt sections
        sections = []
        
        if "system" in template_config:
            sections.append(f"SYSTEM: {template_config['system']}")
        
        if "instructions" in template_config:
            instructions = template_config["instructions"]
            if isinstance(instructions, list):
                sections.append("INSTRUCTIONS:\n" + "\n".join(f"- {i}" for i in instructions))
            else:
                sections.append(f"INSTRUCTIONS: {instructions}")
        
        if "template" in template_config:
            sections.append(template_config["template"].format(**variables))
        
        if "constraints" in template_config:
            constraints = "\n".join(f"- {c}" for c in template_config["constraints"])
            sections.append(f"CONSTRAINTS:\n{constraints}")
        
        return "\n\n".join(sections)

# JSON configuration
json_config = """
{
    "templates": {
        "email_writer": {
            "system": "You are a professional email writer",
            "instructions": [
                "Use appropriate greeting and closing",
                "Keep tone professional but friendly",
                "Be concise and clear"
            ],
            "template": "Write an email to {recipient} about {subject}. Tone: {tone}",
            "constraints": [
                "Maximum 200 words",
                "Include clear call-to-action"
            ]
        },
        "code_explainer": {
            "system": "You are an expert programmer",
            "instructions": "Explain code clearly with examples",
            "template": "Explain this {language} code: {code}",
            "constraints": [
                "Use simple language",
                "Provide analogies"
            ]
        }
    }
}
"""

manager = JSONPromptManager()
manager.load_from_json(json_config)

prompt5 = manager.get_prompt(
    "email_writer",
    recipient="team",
    subject="project deadline extension",
    tone="understanding"
)

print("Example 5 - JSON-Based Prompt Management:")
print(prompt5)