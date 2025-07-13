import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from groq import Groq
import time

class RAGSystemGROQ:
    def __init__(self, knowledge_base_path="data/knowledge_base.json", groq_api_key=None):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None
        self.embeddings = None
        self.index = None
        self.model = None
        self.groq_client = None
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        
        # Load knowledge base
        self.load_knowledge_base()
        
        # Initialize the embedding model (lightweight)
        self.initialize_embedding_model()
        
        # Initialize GROQ client
        self.initialize_groq_client()
        
        # Create document embeddings
        self.create_document_embeddings()
    
    def load_knowledge_base(self):
        """Load the knowledge base from JSON file"""
        if os.path.exists(self.knowledge_base_path):
            with open(self.knowledge_base_path, 'r') as f:
                self.knowledge_base = json.load(f)
        else:
            raise FileNotFoundError(f"Knowledge base not found at {self.knowledge_base_path}")
    
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        print("Loading embedding model...")
        # Using a lightweight model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully!")
    
    def initialize_groq_client(self):
        """Initialize the GROQ client for LLaMA models"""
        print("Initializing GROQ client...")
        if not self.groq_api_key:
            print("Warning: GROQ_API_KEY not found. Please set the environment variable or provide it during initialization.")
            print("You can get a free API key from: https://console.groq.com/")
            self.groq_client = None
        else:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                print("GROQ client initialized successfully!")
            except Exception as e:
                print(f"Error initializing GROQ client: {e}")
                self.groq_client = None
    
    def create_document_embeddings(self):
        """Create embeddings for all documents in the knowledge base"""
        print("Creating document embeddings...")
        
        documents = []
        
        # Convert knowledge base to searchable documents
        kb = self.knowledge_base
        
        # Dataset info
        documents.append(f"Dataset contains {kb['dataset_info']['total_records']} loan records with {kb['dataset_info']['loan_approval_rate']:.1%} approval rate")
        documents.append(f"Average loan amount is ${kb['dataset_info']['average_loan_amount']:.0f}")
        documents.append(f"Average total income is ${kb['dataset_info']['average_income']:.0f}")
        
        # Income analysis
        income_info = kb['feature_insights']['income_analysis']
        documents.append(f"Average applicant income is ${income_info['average_applicant_income']:.0f}")
        documents.append(f"Average coapplicant income is ${income_info['average_coapplicant_income']:.0f}")
        
        # Loan analysis
        loan_info = kb['feature_insights']['loan_analysis']
        documents.append(f"Loan amounts range from ${loan_info['loan_amount_stats']['min']:.0f} to ${loan_info['loan_amount_stats']['max']:.0f}")
        documents.append(f"Most common loan terms are: {list(loan_info['loan_term_distribution'].keys())[:3]}")
        
        # Approval patterns
        approval_patterns = kb['approval_patterns']
        for category, rates in approval_patterns.items():
            for value, rate in rates.items():
                documents.append(f"{category} {value}: {rate:.1%} approval rate")
        
        # Demographic analysis
        demo_info = kb['feature_insights']['demographic_analysis']
        for category, distribution in demo_info.items():
            for value, count in distribution.items():
                documents.append(f"{category} {value}: {count} applicants")
        
        # Create embeddings
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Created embeddings for {len(documents)} documents")
    
    def retrieve_relevant_documents(self, query, top_k=3):
        """Retrieve the most relevant documents for a given query"""
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant documents
        relevant_docs = [self.documents[i] for i in indices[0]]
        
        return relevant_docs, distances[0]
    
    def generate_response_with_groq(self, query, relevant_docs):
        """Generate a response using GROQ API with LLaMA models"""
        if not self.groq_client:
            print("GROQ client not available, using fallback")
            return self.fallback_response(query, relevant_docs)
        
        try:
            # Create context from relevant documents
            context = " ".join(relevant_docs)
            
            # Create a concise prompt for LLaMA
            system_prompt = """You are a friendly loan approval assistant. For greetings, be welcoming and explain your capabilities. 
            For loan questions, provide short, concise answers (2-3 sentences) with key statistics when relevant. 
            Be direct and helpful."""
            
            
            user_prompt = f"""Context: {context}

Question: {query}

Provide a brief, direct answer with key statistics."""

            print(f"Making GROQ API call for query: {query}")
            
            # Use LLaMA-3-8B-8192 model (fast and efficient)
            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                top_p=0.9,
                stream=False
            )
            
            response = completion.choices[0].message.content.strip()
            print(f"GROQ API response received: {response[:100]}...")
            return response
            
        except Exception as e:
            print(f"Error in GROQ API call: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {str(e)}")
            return self.fallback_response(query, relevant_docs)
    
    def fallback_response(self, query, relevant_docs):
        """Fallback response generation when GROQ API is not available"""
        # Clean the query by removing question marks and extra spaces
        query_clean = query.replace('?', '').strip()
        query_lower = query_clean.lower()
        
        print(f"Using fallback response for query: '{query}' (cleaned: '{query_clean}')")
        
        # Handle greetings and general conversation
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']):
            return "Hello! I'm your loan approval assistant. I can answer questions about loan approval patterns and also provide loan prediction features. Check out the prediction tab!"
        
        elif any(word in query_lower for word in ['thanks', 'thank you', 'bye', 'goodbye']):
            return "You're welcome! Feel free to ask more questions about loan approval anytime."
        
        elif any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
            return "I'm a loan approval assistant with multiple features! I can analyze approval patterns, answer questions about loan data, and provide loan prediction capabilities. Check out the prediction tab for personalized loan approval probability."
        
        elif any(word in query_lower for word in ['prediction', 'predict', 'loan prediction', 'prediction system']):
            return "I'm not just a prediction system, but I do provide that feature! Check out the prediction tab where you can input your details and get personalized loan approval probability."
        
        # Loan-related responses
        elif "approval rate" in query_lower:
            approval_rate = self.knowledge_base['dataset_info']['loan_approval_rate']
            return f"Overall approval rate: {approval_rate:.1%}."
        
        elif "average loan" in query_lower or "loan amount" in query_lower:
            avg_loan = self.knowledge_base['dataset_info']['average_loan_amount']
            return f"Average loan amount: ${avg_loan:.0f}."
        
        elif "income" in query_lower:
            avg_income = self.knowledge_base['dataset_info']['average_income']
            return f"Average total income: ${avg_income:.0f}."
        
        elif "credit history" in query_lower:
            credit_rates = self.knowledge_base['approval_patterns']['by_credit_history']
            return f"Good credit: {credit_rates[1]:.1%} approval, poor credit: {credit_rates[0]:.1%} approval."
        
        elif "gender" in query_lower:
            gender_rates = self.knowledge_base['approval_patterns']['by_gender']
            return f"Male: {gender_rates['Male']:.1%} approval, Female: {gender_rates['Female']:.1%} approval."
        
        elif "education" in query_lower:
            edu_rates = self.knowledge_base['approval_patterns']['by_education']
            return f"Graduate: {edu_rates['Graduate']:.1%} approval, Non-graduate: {edu_rates['Not Graduate']:.1%} approval."
        
        elif "property area" in query_lower or "area" in query_lower:
            area_rates = self.knowledge_base['approval_patterns']['by_property_area']
            best_area = max(area_rates, key=area_rates.get)
            return f"{best_area} area: {area_rates[best_area]:.1%} approval rate."
        
        else:
            # Return relevant context or general response
            if relevant_docs:
                return f"Key data: {' '.join(relevant_docs[:1])}"
            else:
                return "I'm here to help with loan approval analysis and prediction! Ask about approval rates, income requirements, or check the prediction tab for personalized loan approval probability."
    
    def answer_question(self, query):
        """Main method to answer a question using RAG with GROQ"""
        print(f"Processing query: {query}")
        
        # Retrieve relevant documents
        relevant_docs, distances = self.retrieve_relevant_documents(query)
        
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        
        # Generate response using GROQ
        response = self.generate_response_with_groq(query, relevant_docs)
        
        # Convert distances to confidence scores safely
        confidence_scores = []
        for distance in distances:
            try:
                confidence = 1.0 / (1.0 + float(distance))
                confidence_scores.append(confidence)
            except (ValueError, TypeError, ZeroDivisionError):
                confidence_scores.append(0.8)  # Default confidence
        
        return {
            'query': query,
            'response': response,
            'relevant_docs': relevant_docs,
            'confidence_scores': confidence_scores
        }

if __name__ == "__main__":
    # Test the RAG system with GROQ
    rag = RAGSystemGROQ()
    
    test_questions = [
        "What is the loan approval rate?",
        "How does income affect loan approval?",
        "What is the average loan amount?",
        "How does credit history impact approval?"
    ]
    
    for question in test_questions:
        result = rag.answer_question(question)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['response']}")
        print("-" * 50) 