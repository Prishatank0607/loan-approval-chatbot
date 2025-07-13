import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the RAG chatbot"""
    
    # GROQ API Configuration
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    GROQ_MODEL: str = "llama3-8b-8192"  # Default LLaMA model
    
    # RAG Configuration
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base.json"
    TOP_K_DOCUMENTS: int = 3
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    
    # UI Configuration
    PAGE_TITLE: str = "Loan Approval RAG Chatbot with GROQ"
    PAGE_ICON: str = "🏦"
    
    @classmethod
    def validate_groq_setup(cls) -> bool:
        """Validate if GROQ API is properly configured"""
        if not cls.GROQ_API_KEY:
            print("⚠️  GROQ_API_KEY not found!")
            print("To use GROQ API with LLaMA models:")
            print("1. Get a free API key from: https://console.groq.com/")
            print("2. Set the environment variable: export GROQ_API_KEY='your-api-key'")
            print("3. Or set it in your shell: GROQ_API_KEY='your-api-key' streamlit run main.py")
            return False
        return True
    
    @classmethod
    def get_groq_api_key(cls) -> Optional[str]:
        """Get GROQ API key with validation"""
        if cls.validate_groq_setup():
            return cls.GROQ_API_KEY
        return None 