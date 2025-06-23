"""
LangChain Prompt Optimization Examples
This file demonstrates various prompt optimization techniques and best practices.
"""

import os
import time
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

class PromptEvaluator:
    """Class to evaluate and compare different prompts."""
    
    def __init__(self, llm):
        self.llm = llm
        self.results = []
    
    def evaluate_prompt(self, prompt: str, test_cases: List[Dict], metrics: List[str]) -> Dict:
        """Evaluate a prompt against test cases and metrics."""
        results = {
            'prompt': prompt,
            'test_cases': [],
            'overall_score': 0,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            case_result = {
                'input': test_case['input'],
                'expected': test_case.get('expected', None),
                'response': None,
                'score': 0,
                'metrics': {}
            }
            
            try:
                # Execute the prompt
                response = self.llm.invoke(prompt.format(**test_case))
                case_result['response'] = response.content
                
                # Calculate metrics
                for metric in metrics:
                    case_result['metrics'][metric] = self._calculate_metric(
                        metric, case_result['response'], test_case
                    )
                
                # Calculate overall score for this case
                case_result['score'] = sum(case_result['metrics'].values()) / len(metrics)
                
            except Exception as e:
                case_result['error'] = str(e)
                case_result['score'] = 0
            
            results['test_cases'].append(case_result)
        
        results['execution_time'] = time.time() - start_time
        results['overall_score'] = sum(case['score'] for case in results['test_cases']) / len(results['test_cases'])
        
        return results
    
    def _calculate_metric(self, metric: str, response: str, test_case: Dict) -> float:
        """Calculate a specific metric for the response."""
        if metric == 'length':
            return min(len(response) / 100, 1.0)  # Normalize to 0-1
        elif metric == 'relevance':
            # Simple keyword matching for relevance
            keywords = test_case.get('keywords', [])
            if keywords:
                matches = sum(1 for keyword in keywords if keyword.lower() in response.lower())
                return matches / len(keywords)
            return 0.5
        elif metric == 'clarity':
            # Simple clarity metric based on sentence structure
            sentences = response.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            return min(avg_sentence_length / 20, 1.0)  # Prefer moderate sentence length
        elif metric == 'completeness':
            # Check if response addresses all expected aspects
            expected_aspects = test_case.get('expected_aspects', [])
            if expected_aspects:
                addressed = sum(1 for aspect in expected_aspects if aspect.lower() in response.lower())
                return addressed / len(expected_aspects)
            return 0.5
        else:
            return 0.5

def ab_testing_example():
    """Demonstrate A/B testing of different prompt variations."""
    print("=== A/B Testing Example ===")
    
    # Define different prompt variations
    prompt_variations = {
        "direct": "Explain {topic} in simple terms.",
        "detailed": "Please provide a comprehensive explanation of {topic} that covers the main concepts, examples, and practical applications.",
        "question_based": "What is {topic}? How does it work? Why is it important?",
        "role_based": "You are a teacher explaining {topic} to a student. Make it easy to understand with examples.",
        "step_by_step": "Let's break down {topic} step by step. First, explain what it is, then how it works, and finally why it matters."
    }
    
    # Define test cases
    test_cases = [
        {
            'topic': 'machine learning',
            'keywords': ['algorithm', 'data', 'prediction', 'training'],
            'expected_aspects': ['definition', 'examples', 'applications']
        },
        {
            'topic': 'blockchain technology',
            'keywords': ['distributed', 'ledger', 'cryptocurrency', 'security'],
            'expected_aspects': ['concept', 'benefits', 'use cases']
        },
        {
            'topic': 'artificial intelligence',
            'keywords': ['intelligence', 'automation', 'learning', 'decision'],
            'expected_aspects': ['definition', 'types', 'impact']
        }
    ]
    
    # Define metrics to evaluate
    metrics = ['relevance', 'clarity', 'completeness', 'length']
    
    # Create evaluator
    evaluator = PromptEvaluator(llm)
    
    # Test each prompt variation
    results = {}
    for name, prompt in prompt_variations.items():
        print(f"\nTesting {name} prompt...")
        results[name] = evaluator.evaluate_prompt(prompt, test_cases, metrics)
    
    # Compare results
    print("\n=== A/B Testing Results ===")
    for name, result in results.items():
        print(f"\n{name.upper()} Prompt:")
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Average Response Length: {sum(len(case['response']) for case in result['test_cases']) / len(result['test_cases']):.0f} chars")
    
    # Find best performing prompt
    best_prompt = max(results.items(), key=lambda x: x[1]['overall_score'])
    print(f"\nBest Performing Prompt: {best_prompt[0]} (Score: {best_prompt[1]['overall_score']:.3f})")

def prompt_iteration_example():
    """Demonstrate iterative prompt improvement."""
    print("\n=== Prompt Iteration Example ===")
    
    # Initial prompt
    initial_prompt = "Write a summary of {text}."
    
    # Test case
    test_case = {
        'text': "Artificial intelligence is transforming industries worldwide. Machine learning algorithms are being used to automate processes, improve decision-making, and create new products and services. Companies are investing heavily in AI research and development, and the technology is expected to continue growing rapidly in the coming years."
    }
    
    # Define evaluation criteria
    evaluation_criteria = {
        'length': 'Should be 2-3 sentences',
        'clarity': 'Should be easy to understand',
        'completeness': 'Should cover main points',
        'structure': 'Should be well-organized'
    }
    
    # Iterative improvements
    prompt_versions = [
        {
            'version': 1,
            'prompt': initial_prompt,
            'issues': 'Too vague, no specific requirements'
        },
        {
            'version': 2,
            'prompt': "Write a 2-3 sentence summary of {text} that captures the main points.",
            'issues': 'Better but could be more specific about structure'
        },
        {
            'version': 3,
            'prompt': "Summarize {text} in 2-3 clear, well-structured sentences. Focus on the key concepts and their significance.",
            'issues': 'Good structure, could add more guidance'
        },
        {
            'version': 4,
            'prompt': """Summarize the following text in 2-3 sentences:

{text}

Your summary should:
- Capture the main ideas and key points
- Be clear and easy to understand
- Highlight the significance or impact
- Use active voice and concise language""",
            'issues': 'Comprehensive but might be too verbose'
        },
        {
            'version': 5,
            'prompt': "Create a concise 2-3 sentence summary of {text}. Focus on the main topic, key developments, and their importance.",
            'issues': 'Balanced approach - clear, specific, and concise'
        }
    ]
    
    # Test each version
    evaluator = PromptEvaluator(llm)
    test_cases = [test_case]
    metrics = ['length', 'clarity', 'completeness']
    
    print("Testing prompt iterations...")
    for version_info in prompt_versions:
        print(f"\n--- Version {version_info['version']} ---")
        print(f"Issues addressed: {version_info['issues']}")
        
        result = evaluator.evaluate_prompt(version_info['prompt'], test_cases, metrics)
        
        print(f"Score: {result['overall_score']:.3f}")
        print(f"Response: {result['test_cases'][0]['response']}")
    
    print("\nPrompt iteration completed!")

def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    print("\n=== Performance Optimization Example ===")
    
    # Test different prompt optimization strategies
    strategies = {
        'verbose': """You are a helpful assistant. Please analyze the following text and provide a comprehensive response.

Text: {text}

Please consider:
1. The main topic and key themes
2. Important details and supporting information
3. The overall message or conclusion
4. Any implications or significance

Provide a detailed analysis that covers all these aspects thoroughly.""",
        
        'concise': "Analyze {text} and provide key insights.",
        
        'structured': """Analyze this text: {text}

Provide:
- Main topic
- Key points
- Conclusion""",
        
        'contextual': "Given the context of {text}, what are the main insights?",
        
        'instructional': "Extract the main ideas from: {text}"
    }
    
    # Test case
    test_case = {
        'text': "The rapid advancement of renewable energy technologies has made solar and wind power increasingly cost-competitive with traditional fossil fuels. This shift is driving significant changes in global energy markets and investment patterns."
    }
    
    # Performance metrics
    evaluator = PromptEvaluator(llm)
    test_cases = [test_case]
    metrics = ['length', 'clarity', 'completeness']
    
    print("Testing performance optimization strategies...")
    results = {}
    
    for strategy_name, prompt in strategies.items():
        print(f"\n--- {strategy_name.upper()} Strategy ---")
        
        result = evaluator.evaluate_prompt(prompt, test_cases, metrics)
        results[strategy_name] = result
        
        print(f"Score: {result['overall_score']:.3f}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Response Length: {len(result['test_cases'][0]['response'])} chars")
        print(f"Response: {result['test_cases'][0]['response'][:100]}...")
    
    # Find optimal strategy
    optimal_strategy = max(results.items(), key=lambda x: x[1]['overall_score'] / x[1]['execution_time'])
    print(f"\nOptimal Strategy: {optimal_strategy[0]} (Efficiency: {optimal_strategy[1]['overall_score'] / optimal_strategy[1]['execution_time']:.3f})")

def prompt_engineering_best_practices():
    """Demonstrate prompt engineering best practices."""
    print("\n=== Prompt Engineering Best Practices ===")
    
    # Examples of good vs bad prompts
    examples = [
        {
            'category': 'Clarity',
            'bad': "Do something with this data: {data}",
            'good': "Analyze the following dataset and identify the top 3 trends: {data}",
            'explanation': 'Specific instructions are better than vague requests'
        },
        {
            'category': 'Context',
            'bad': "Write about AI",
            'good': "You are a technology journalist. Write a 300-word article about recent developments in AI for a general audience.",
            'explanation': 'Providing context and role improves output quality'
        },
        {
            'category': 'Constraints',
            'bad': "Explain machine learning",
            'good': "Explain machine learning in 3 sentences using simple analogies that a high school student would understand.",
            'explanation': 'Adding constraints helps focus the response'
        },
        {
            'category': 'Examples',
            'bad': "Classify this text: {text}",
            'good': """Classify this text as positive, negative, or neutral:

Examples:
- "I love this product!" → positive
- "This is terrible." → negative
- "It's okay." → neutral

Text to classify: {text}""",
            'explanation': 'Few-shot examples improve accuracy'
        },
        {
            'category': 'Output Format',
            'bad': "List the benefits of remote work",
            'good': """List the benefits of remote work in this format:

1. Benefit Name
   - Description
   - Example

2. Benefit Name
   - Description
   - Example""",
            'explanation': 'Specifying output format ensures consistency'
        }
    ]
    
    print("Prompt Engineering Best Practices:")
    for example in examples:
        print(f"\n--- {example['category']} ---")
        print(f"❌ Bad: {example['bad']}")
        print(f"✅ Good: {example['good']}")
        print(f"💡 Why: {example['explanation']}")
        
        # Test the good prompt
        if '{text}' in example['good']:
            test_input = "This movie was absolutely fantastic!"
            test_prompt = example['good'].format(text=test_input)
            response = llm.invoke(test_prompt)
            print(f"📝 Result: {response.content[:100]}...")
        elif '{data}' in example['good']:
            test_data = "Sales: 100, 150, 200, 250, 300 (last 5 months)"
            test_prompt = example['good'].format(data=test_data)
            response = llm.invoke(test_prompt)
            print(f"📝 Result: {response.content[:100]}...")

def prompt_validation_example():
    """Demonstrate prompt validation and testing."""
    print("\n=== Prompt Validation Example ===")
    
    # Define a prompt to validate
    prompt = """You are a helpful coding assistant. Review this code and provide feedback:

{code}

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement"""

    # Define validation test cases
    validation_cases = [
        {
            'name': 'Simple Function',
            'code': 'def add(a, b): return a + b',
            'expected_aspects': ['function', 'simple', 'correct']
        },
        {
            'name': 'Complex Algorithm',
            'code': '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            'expected_aspects': ['recursion', 'performance', 'optimization']
        },
        {
            'name': 'Error-Prone Code',
            'code': '''def divide_numbers(a, b):
    return a / b''',
            'expected_aspects': ['error handling', 'division by zero', 'defensive programming']
        }
    ]
    
    # Validate the prompt
    evaluator = PromptEvaluator(llm)
    metrics = ['completeness', 'relevance', 'clarity']
    
    print("Validating prompt with different code examples...")
    results = evaluator.evaluate_prompt(prompt, validation_cases, metrics)
    
    print(f"\nOverall Validation Score: {results['overall_score']:.3f}")
    
    for i, case in enumerate(results['test_cases']):
        print(f"\n--- {validation_cases[i]['name']} ---")
        print(f"Score: {case['score']:.3f}")
        print(f"Response: {case['response'][:200]}...")
        
        # Check if expected aspects are covered
        response_lower = case['response'].lower()
        for aspect in validation_cases[i]['expected_aspects']:
            covered = aspect.lower() in response_lower
            print(f"  {aspect}: {'✅' if covered else '❌'}")

def main():
    """Run all prompt optimization examples."""
    print("LangChain Prompt Optimization Examples\n")
    print("=" * 50)
    
    try:
        ab_testing_example()
        prompt_iteration_example()
        performance_optimization_example()
        prompt_engineering_best_practices()
        prompt_validation_example()
        
        print("\nAll prompt optimization examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your OpenAI API key in the .env file.")

if __name__ == "__main__":
    main() 