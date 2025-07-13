import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

class RAGSystem:
    def __init__(self, knowledge_base_path="data/knowledge_base.json"):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None
        self.embeddings = None
        self.index = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Load knowledge base
        self.load_knowledge_base()
        
        # Initialize the embedding model (lightweight)
        self.initialize_embedding_model()
        
        # Initialize the generation model (lightweight)
        self.initialize_generation_model()
        
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
    
    def initialize_generation_model(self):
        """Initialize the text generation model"""
        print("Loading generation model...")
        try:
            # Using a lightweight model for text generation
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.generator = pipeline("text-generation", model=model_name, tokenizer=self.tokenizer)
            print("Generation model loaded successfully!")
        except Exception as e:
            print(f"Error loading generation model: {e}")
            print("Using fallback response generation...")
            self.generator = None
    
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
    
    def generate_response(self, query, relevant_docs):
        """Generate a response using the retrieved documents"""
        # Create context from relevant documents
        context = " ".join(relevant_docs)
        
        # Create prompt
        prompt = f"Context about loan approval data: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        if self.generator:
            try:
                # Generate response using the model
                response = self.generator(prompt, max_length=200, num_return_sequences=1, 
                                       temperature=0.7, do_sample=True)
                generated_text = response[0]['generated_text']
                
                # Extract the answer part (after "Answer:")
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text.split("Question:")[-1].strip()
                
                return answer
            except Exception as e:
                print(f"Error in model generation: {e}")
                return self.fallback_response(query, relevant_docs)
        else:
            return self.fallback_response(query, relevant_docs)
    
    def fallback_response(self, query, relevant_docs):
        """Fallback response generation when model is not available"""
        query_lower = query.lower()
        
        # Simple rule-based responses
        if "approval rate" in query_lower:
            approval_rate = self.knowledge_base['dataset_info']['loan_approval_rate']
            return f"The overall loan approval rate is {approval_rate:.1%}."
        
        elif "average loan" in query_lower or "loan amount" in query_lower:
            avg_loan = self.knowledge_base['dataset_info']['average_loan_amount']
            return f"The average loan amount is ${avg_loan:.0f}."
        
        elif "income" in query_lower:
            avg_income = self.knowledge_base['dataset_info']['average_income']
            return f"The average total income of applicants is ${avg_income:.0f}."
        
        elif "credit history" in query_lower:
            credit_rates = self.knowledge_base['approval_patterns']['by_credit_history']
            return f"Applicants with good credit history have a {credit_rates[1]:.1%} approval rate, while those with poor credit have a {credit_rates[0]:.1%} approval rate."
        
        elif "gender" in query_lower:
            gender_rates = self.knowledge_base['approval_patterns']['by_gender']
            return f"Male applicants have a {gender_rates['Male']:.1%} approval rate, while female applicants have a {gender_rates['Female']:.1%} approval rate."
        
        elif "education" in query_lower:
            edu_rates = self.knowledge_base['approval_patterns']['by_education']
            return f"Graduate applicants have a {edu_rates['Graduate']:.1%} approval rate, while non-graduates have a {edu_rates['Not Graduate']:.1%} approval rate."
        
        elif "property area" in query_lower or "area" in query_lower:
            area_rates = self.knowledge_base['approval_patterns']['by_property_area']
            best_area = max(area_rates, key=area_rates.get)
            return f"The {best_area} property area has the highest approval rate at {area_rates[best_area]:.1%}."
        
        else:
            # Return relevant context
            return f"Based on the loan approval data: {' '.join(relevant_docs[:2])}"
    
    def answer_question(self, query):
        """Main method to answer a question using RAG"""
        print(f"Processing query: {query}")
        
        # Retrieve relevant documents
        relevant_docs, distances = self.retrieve_relevant_documents(query)
        
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        return {
            'query': query,
            'response': response,
            'relevant_docs': relevant_docs,
            'confidence_scores': 1.0 / (1.0 + distances)  # Convert distances to confidence scores
        }

if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    
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