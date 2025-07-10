"""
Real-Time LangChain Vector Store Usage Examples
===============================================

This file contains practical, production-ready examples of using LangChain vector stores
in real-time applications including document search, chatbots, recommendation systems,
and more.
"""

import os
import time
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from threading import Thread
import logging

# LangChain imports
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. REAL-TIME DOCUMENT SEARCH SYSTEM
# ==============================================================================

class RealTimeDocumentSearchSystem:
    """
    A real-time document search system that can handle dynamic document updates
    and provide instant search results.
    """
    
    def __init__(self, persist_directory: str = "./realtime_search_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Text splitter for processing documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Document tracking
        self.document_index = {}
        self.last_update = datetime.now()
        
    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        if os.path.exists(self.persist_directory):
            try:
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
                return vector_store
            except Exception as e:
                logger.warning(f"Could not load existing store: {e}")
        
        # Create new vector store
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        logger.info("Created new vector store")
        return vector_store
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a single document to the search system"""
        doc_id = metadata.get('id', f"doc_{int(time.time() * 1000)}")
        
        # Split document into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                'chunk_id': f"{doc_id}_chunk_{i}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'timestamp': datetime.now().isoformat()
            }
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
        
        # Update document index
        self.document_index[doc_id] = {
            'content': content,
            'metadata': metadata,
            'chunks': len(chunks),
            'added_at': datetime.now().isoformat()
        }
        
        self.last_update = datetime.now()
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        
        return doc_id
    
    def search_documents(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search for documents with optional filters"""
        try:
            if filters:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filters
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score),
                    'relevance': 1.0 - score  # Convert distance to relevance
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]):
        """Update an existing document"""
        # Remove old document
        self.remove_document(doc_id)
        
        # Add updated document
        new_metadata['id'] = doc_id
        return self.add_document(new_content, new_metadata)
    
    def remove_document(self, doc_id: str):
        """Remove a document and all its chunks"""
        try:
            # Remove from vector store (filter by document ID)
            self.vector_store.delete(filter={"id": doc_id})
            
            # Remove from document index
            if doc_id in self.document_index:
                del self.document_index[doc_id]
            
            self.vector_store.persist()
            logger.info(f"Removed document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            collection = self.vector_store._collection
            total_chunks = collection.count()
        except:
            total_chunks = 0
        
        return {
            'total_documents': len(self.document_index),
            'total_chunks': total_chunks,
            'last_update': self.last_update.isoformat(),
            'vector_store_type': type(self.vector_store).__name__
        }


# ==============================================================================
# 2. REAL-TIME CHATBOT WITH DYNAMIC KNOWLEDGE
# ==============================================================================

class RealTimeChatbot:
    """
    A real-time chatbot that can learn from conversations and update its knowledge base
    """
    
    def __init__(self, knowledge_base_path: str = "./chatbot_kb"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="microsoft/DialoGPT-medium",
            task="text-generation",
            model_kwargs={"temperature": 0.7, "max_length": 200}
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_knowledge_base()
        
        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create retrieval chain
        self.qa_chain = self._create_qa_chain()
        
        # Conversation tracking
        self.conversation_history = []
        self.user_feedback = []
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base"""
        if os.path.exists(self.knowledge_base_path):
            try:
                return Chroma(
                    persist_directory=self.knowledge_base_path,
                    embedding_function=self.embeddings
                )
            except:
                pass
        
        # Create with initial knowledge
        initial_knowledge = [
            Document(
                page_content="I am an AI assistant created to help answer questions and have conversations.",
                metadata={"type": "system", "topic": "identity"}
            ),
            Document(
                page_content="I can help with various topics including technology, science, and general knowledge.",
                metadata={"type": "system", "topic": "capabilities"}
            )
        ]
        
        vector_store = Chroma.from_documents(
            initial_knowledge,
            self.embeddings,
            persist_directory=self.knowledge_base_path
        )
        
        return vector_store
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def chat(self, user_input: str, user_id: str = "user") -> Dict:
        """Process user input and return response"""
        start_time = time.time()
        
        try:
            # Get response from chain
            result = self.qa_chain({"question": user_input})
            
            response = {
                "user_input": user_input,
                "bot_response": result["answer"],
                "sources": [doc.page_content for doc in result["source_documents"]],
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time
            }
            
            # Track conversation
            self.conversation_history.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "user_input": user_input,
                "bot_response": "I'm sorry, I encountered an error. Please try again.",
                "sources": [],
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def learn_from_conversation(self, conversation_text: str, topic: str):
        """Add new knowledge from conversation"""
        new_doc = Document(
            page_content=conversation_text,
            metadata={
                "type": "learned",
                "topic": topic,
                "learned_at": datetime.now().isoformat()
            }
        )
        
        self.vector_store.add_documents([new_doc])
        self.vector_store.persist()
        
        # Recreate chain with updated knowledge
        self.qa_chain = self._create_qa_chain()
        
        logger.info(f"Learned new information about {topic}")
    
    def add_user_feedback(self, conversation_id: int, rating: int, comment: str = ""):
        """Add user feedback for improving responses"""
        feedback = {
            "conversation_id": conversation_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        self.user_feedback.append(feedback)
        
        # Learn from positive feedback
        if rating >= 4 and conversation_id < len(self.conversation_history):
            conv = self.conversation_history[conversation_id]
            self.learn_from_conversation(
                f"Q: {conv['user_input']}\nA: {conv['bot_response']}",
                "user_approved"
            )
    
    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        avg_response_time = sum(c.get("response_time", 0) for c in self.conversation_history) / len(self.conversation_history)
        
        return {
            "total_conversations": len(self.conversation_history),
            "average_response_time": avg_response_time,
            "total_feedback": len(self.user_feedback),
            "knowledge_base_size": self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else 0
        }


# ==============================================================================
# 3. REAL-TIME RECOMMENDATION SYSTEM
# ==============================================================================

class RealTimeRecommendationSystem:
    """
    A real-time recommendation system using vector similarity
    """
    
    def __init__(self, db_path: str = "./recommendations_db"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector stores for different item types
        self.item_store = self._initialize_store("items")
        self.user_profile_store = self._initialize_store("user_profiles")
        
        # User interaction tracking
        self.user_interactions = {}
        self.item_popularity = {}
    
    def _initialize_store(self, store_type: str):
        """Initialize a vector store for specific type"""
        store_path = f"{self.db_path}/{store_type}"
        
        if os.path.exists(store_path):
            try:
                return Chroma(
                    persist_directory=store_path,
                    embedding_function=self.embeddings
                )
            except:
                pass
        
        return Chroma(
            persist_directory=store_path,
            embedding_function=self.embeddings
        )
    
    def add_item(self, item_id: str, title: str, description: str, 
                 category: str, tags: List[str] = None) -> str:
        """Add an item to the recommendation system"""
        
        # Create item text for embedding
        item_text = f"Title: {title}\nDescription: {description}\nCategory: {category}"
        if tags:
            item_text += f"\nTags: {', '.join(tags)}"
        
        # Create document
        item_doc = Document(
            page_content=item_text,
            metadata={
                "item_id": item_id,
                "title": title,
                "category": category,
                "tags": tags or [],
                "added_at": datetime.now().isoformat()
            }
        )
        
        # Add to vector store
        self.item_store.add_documents([item_doc])
        self.item_store.persist()
        
        # Initialize popularity tracking
        self.item_popularity[item_id] = 0
        
        logger.info(f"Added item {item_id}: {title}")
        return item_id
    
    def update_user_profile(self, user_id: str, interests: List[str], 
                           preferences: Dict[str, Any]):
        """Update user profile for better recommendations"""
        
        # Create user profile text
        profile_text = f"User interests: {', '.join(interests)}\n"
        for key, value in preferences.items():
            profile_text += f"{key}: {value}\n"
        
        # Create or update user profile document
        profile_doc = Document(
            page_content=profile_text,
            metadata={
                "user_id": user_id,
                "interests": interests,
                "preferences": preferences,
                "updated_at": datetime.now().isoformat()
            }
        )
        
        # Remove old profile if exists
        try:
            self.user_profile_store.delete(filter={"user_id": user_id})
        except:
            pass
        
        # Add new profile
        self.user_profile_store.add_documents([profile_doc])
        self.user_profile_store.persist()
        
        logger.info(f"Updated profile for user {user_id}")
    
    def record_interaction(self, user_id: str, item_id: str, interaction_type: str, 
                          rating: Optional[float] = None):
        """Record user interaction with an item"""
        
        interaction = {
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction_type,  # view, like, purchase, etc.
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update user interactions
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        self.user_interactions[user_id].append(interaction)
        
        # Update item popularity
        if item_id in self.item_popularity:
            weight = {"view": 1, "like": 2, "purchase": 3}.get(interaction_type, 1)
            self.item_popularity[item_id] += weight
        
        logger.info(f"Recorded {interaction_type} interaction: {user_id} -> {item_id}")
    
    def get_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[Dict]:
        """Get recommendations for a user"""
        
        try:
            # Get user profile
            user_profiles = self.user_profile_store.similarity_search(
                f"user_id:{user_id}", k=1, filter={"user_id": user_id}
            )
            
            if not user_profiles:
                # No profile found, return popular items
                return self._get_popular_items(num_recommendations)
            
            user_profile = user_profiles[0]
            
            # Get similar items based on user profile
            similar_items = self.item_store.similarity_search_with_score(
                user_profile.page_content, k=num_recommendations * 2
            )
            
            # Filter out items user has already interacted with
            user_item_history = set()
            if user_id in self.user_interactions:
                user_item_history = {
                    interaction["item_id"] 
                    for interaction in self.user_interactions[user_id]
                }
            
            recommendations = []
            for item_doc, score in similar_items:
                item_id = item_doc.metadata["item_id"]
                
                if item_id not in user_item_history:
                    # Calculate recommendation score
                    popularity_score = self.item_popularity.get(item_id, 0)
                    similarity_score = 1.0 - score  # Convert distance to similarity
                    
                    # Combine scores
                    final_score = 0.7 * similarity_score + 0.3 * min(popularity_score / 10, 1.0)
                    
                    recommendations.append({
                        "item_id": item_id,
                        "title": item_doc.metadata["title"],
                        "category": item_doc.metadata["category"],
                        "similarity_score": similarity_score,
                        "popularity_score": popularity_score,
                        "final_score": final_score,
                        "reason": "Based on your interests"
                    })
                
                if len(recommendations) >= num_recommendations:
                    break
            
            # Sort by final score
            recommendations.sort(key=lambda x: x["final_score"], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self._get_popular_items(num_recommendations)
    
    def _get_popular_items(self, num_items: int) -> List[Dict]:
        """Get popular items as fallback"""
        
        # Sort items by popularity
        popular_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_items]
        
        recommendations = []
        for item_id, popularity in popular_items:
            # Get item details
            item_docs = self.item_store.similarity_search(
                f"item_id:{item_id}", k=1, filter={"item_id": item_id}
            )
            
            if item_docs:
                item_doc = item_docs[0]
                recommendations.append({
                    "item_id": item_id,
                    "title": item_doc.metadata["title"],
                    "category": item_doc.metadata["category"],
                    "popularity_score": popularity,
                    "final_score": popularity,
                    "reason": "Popular item"
                })
        
        return recommendations


# ==============================================================================
# 4. REAL-TIME CONTENT MONITORING SYSTEM
# ==============================================================================

class RealTimeContentMonitor:
    """
    A real-time content monitoring system that can detect similar content,
    track changes, and alert on specific patterns
    """
    
    def __init__(self, db_path: str = "./content_monitor_db"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vector_store = self._initialize_store()
        self.alert_thresholds = {
            "similarity": 0.9,  # Alert if content is too similar
            "sentiment": -0.5,  # Alert if content is too negative
            "spam": 0.8         # Alert if content looks like spam
        }
        
        self.alerts = []
        self.content_history = []
        
        # Background monitoring
        self.is_monitoring = False
        self.monitor_thread = None
    
    def _initialize_store(self):
        """Initialize the content monitoring store"""
        if os.path.exists(self.db_path):
            try:
                return Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
            except:
                pass
        
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
    
    def add_content(self, content_id: str, content_text: str, 
                   source: str, content_type: str = "text") -> Dict:
        """Add content to monitor"""
        
        # Create content document
        content_doc = Document(
            page_content=content_text,
            metadata={
                "content_id": content_id,
                "source": source,
                "content_type": content_type,
                "timestamp": datetime.now().isoformat(),
                "length": len(content_text)
            }
        )
        
        # Check for similar content
        similar_content = self.vector_store.similarity_search_with_score(
            content_text, k=3
        )
        
        analysis_result = {
            "content_id": content_id,
            "added_at": datetime.now().isoformat(),
            "similar_content": [],
            "alerts": []
        }
        
        # Analyze similarity
        for doc, score in similar_content:
            if score < (1.0 - self.alert_thresholds["similarity"]):
                similarity_alert = {
                    "type": "similarity",
                    "message": f"Content similar to {doc.metadata['content_id']}",
                    "similarity_score": 1.0 - score,
                    "original_content_id": doc.metadata["content_id"]
                }
                analysis_result["alerts"].append(similarity_alert)
                self.alerts.append(similarity_alert)
            
            analysis_result["similar_content"].append({
                "content_id": doc.metadata["content_id"],
                "similarity_score": 1.0 - score
            })
        
        # Add to vector store
        self.vector_store.add_documents([content_doc])
        self.vector_store.persist()
        
        # Add to history
        self.content_history.append({
            "content_id": content_id,
            "content_text": content_text[:200] + "..." if len(content_text) > 200 else content_text,
            "source": source,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Added content {content_id} from {source}")
        return analysis_result
    
    def detect_content_patterns(self, pattern_query: str, threshold: float = 0.8) -> List[Dict]:
        """Detect content matching specific patterns"""
        
        results = self.vector_store.similarity_search_with_score(
            pattern_query, k=50
        )
        
        matches = []
        for doc, score in results:
            if (1.0 - score) >= threshold:
                matches.append({
                    "content_id": doc.metadata["content_id"],
                    "source": doc.metadata["source"],
                    "similarity_score": 1.0 - score,
                    "content_preview": doc.page_content[:100] + "...",
                    "timestamp": doc.metadata["timestamp"]
                })
        
        return matches
    
    def monitor_content_stream(self, check_interval: int = 60):
        """Monitor content stream for patterns and alerts"""
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Check for spam patterns
                    spam_patterns = [
                        "buy now", "limited time offer", "click here",
                        "free money", "guaranteed", "act now"
                    ]
                    
                    for pattern in spam_patterns:
                        matches = self.detect_content_patterns(pattern, 0.7)
                        
                        for match in matches:
                            spam_alert = {
                                "type": "spam",
                                "message": f"Potential spam detected: {pattern}",
                                "content_id": match["content_id"],
                                "pattern": pattern,
                                "timestamp": datetime.now().isoformat()
                            }
                            self.alerts.append(spam_alert)
                    
                    # Sleep until next check
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    time.sleep(check_interval)
        
        self.is_monitoring = True
        self.monitor_thread = Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started content monitoring")
    
    def stop_monitoring(self):
        """Stop content monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped content monitoring")
    
    def get_alerts(self, alert_type: str = None, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        alerts = self.alerts
        
        if alert_type:
            alerts = [alert for alert in alerts if alert["type"] == alert_type]
        
        return alerts[-limit:]
    
    def get_content_stats(self) -> Dict:
        """Get content monitoring statistics"""
        total_content = len(self.content_history)
        total_alerts = len(self.alerts)
        
        alert_types = {}
        for alert in self.alerts:
            alert_type = alert["type"]
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        return {
            "total_content_monitored": total_content,
            "total_alerts": total_alerts,
            "alert_breakdown": alert_types,
            "monitoring_active": self.is_monitoring
        }


# ==============================================================================
# 5. DEMO USAGE EXAMPLES
# ==============================================================================

def demo_document_search():
    """Demo the real-time document search system"""
    print("=== Real-Time Document Search Demo ===")
    
    # Initialize system
    search_system = RealTimeDocumentSearchSystem()
    
    # Add some sample documents
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
            "metadata": {"title": "Python Programming", "category": "programming", "author": "Tech Writer"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "metadata": {"title": "Machine Learning Basics", "category": "AI", "author": "Data Scientist"}
        },
        {
            "content": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
            "metadata": {"title": "Climate Change Overview", "category": "environment", "author": "Environmental Scientist"}
        }
    ]
    
    # Add documents
    for doc in documents:
        doc_id = search_system.add_document(doc["content"], doc["metadata"])
        print(f"Added document: {doc_id}")
    
    # Perform searches
    print("\nSearching for 'programming languages'...")
    results = search_system.search_documents("programming languages", k=2)
    for result in results:
        print(f"- {result['metadata']['title']} (Score: {result['relevance']:.3f})")
        print(f"  {result['content'][:100]}...")
    
    # Search with filters
    print("\nSearching for 'learning' in AI category...")
    results = search_system.search_documents("learning", k=2, filters={"category": "AI"})
    for result in results:
        print(f"- {result['metadata']['title']} (Score: {result['relevance']:.3f})")
    
    # Show system stats
    stats = search_system.get_system_stats()
    print(f"\nSystem Stats: {stats}")


def demo_chatbot():
    """Demo the real-time chatbot"""
    print("\n=== Real-Time Chatbot Demo ===")
    
    # Initialize chatbot
    chatbot = RealTimeChatbot()
    
    # Add some knowledge
    chatbot.learn_from_conversation(
        "Python is excellent for data science because of libraries like pandas, numpy, and scikit-learn.",
        "python_data_science"
    )
    
    # Simulate conversation
    questions = [
        "What programming language is good for data science?",
        "Can you tell me more about Python?",
        "What libraries are useful for data science?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = chatbot.chat(question)
        print(f"Bot: {response['bot_response']}")
        print(f"Response time: {response['response_time']:.2f}s")
    
    # Show conversation stats
    stats = chatbot.get_conversation_stats()
    print(f"\nChatbot Stats: {stats}")


def demo_recommendations():
    """Demo the real-time recommendation system"""
    print("\n=== Real-Time Recommendation Demo ===")
    
    # Initialize recommendation system
    rec_system = RealTimeRecommendationSystem()
    
    # Add some items
    items = [
        ("book_1", "Python Programming Guide", "Complete guide to Python programming", "books", ["python", "programming"]),
        ("book_2", "Machine Learning Handbook", "Comprehensive ML resource", "books", ["