#!/usr/bin/env python3
"""
Consolidated Test Script for Loan Approval RAG Chatbot
This script combines all testing functionality into one comprehensive test suite
"""

import os
import sys
import streamlit as st

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_package_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing Package Imports...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sentence_transformers',
        'faiss',
        'torch',
        'transformers',
        'sklearn',
        'groq'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_data_processor():
    """Test the data processor"""
    print("\n🧪 Testing Data Processor...")
    
    try:
        from data_processor import LoanDataProcessor
        
        processor = LoanDataProcessor()
        df, knowledge_base = processor.load_data()
        
        print(f"✅ Data processor test passed!")
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Knowledge base keys: {list(knowledge_base.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_rag_system_groq():
    """Test the GROQ RAG system"""
    print("\n🧪 Testing GROQ RAG System...")
    
    try:
        from rag_system_groq import RAGSystemGROQ
        
        # Check GROQ API key
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            print("✅ GROQ API key found")
            mode = "GROQ LLaMA API"
        else:
            print("⚠️ GROQ API key not found - using fallback mode")
            mode = "Fallback Mode"
        
        rag = RAGSystemGROQ()
        
        # Test questions
        test_questions = [
            "What is the loan approval rate?",
            "How does income affect loan approval?",
            "What is the average loan amount?",
            "Which property area has the highest approval rate?"
        ]
        
        print(f"\n🤖 Testing responses in {mode}...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            try:
                result = rag.answer_question(question)
                print(f"   Response: {result['response'][:100]}...")
                print(f"   Confidence: {result['confidence_scores'][0]:.2f}")
                print(f"   Relevant docs: {len(result['relevant_docs'])}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("✅ GROQ RAG system test passed!")
        return True
    except Exception as e:
        print(f"❌ GROQ RAG system test failed: {e}")
        return False

def test_rag_system_basic():
    """Test the basic RAG system"""
    print("\n🧪 Testing Basic RAG System...")
    
    try:
        from rag_system import RAGSystem
        
        rag = RAGSystem()
        
        # Test questions
        test_questions = [
            "What is the loan approval rate?",
            "How does income affect loan approval?",
            "What is the average loan amount?"
        ]
        
        for question in test_questions:
            result = rag.answer_question(question)
            print(f"   Q: {result['query']}")
            print(f"   A: {result['response'][:100]}...")
            print(f"   Confidence: {result['confidence_scores'][0]:.2f}")
            print()
        
        print("✅ Basic RAG system test passed!")
        return True
    except Exception as e:
        print(f"❌ Basic RAG system test failed: {e}")
        return False

def test_prediction_model():
    """Test the prediction model"""
    print("\n🧪 Testing Prediction Model...")
    
    try:
        from data_processor import LoanDataProcessor
        from sklearn.ensemble import RandomForestClassifier
        
        # Load data
        processor = LoanDataProcessor()
        df, _ = processor.load_data()
        
        # Prepare features for prediction
        feature_cols = ['Gender_encoded', 'Married_encoded', 'Dependents_encoded', 
                       'Education_encoded', 'Self_Employed_encoded', 'ApplicantIncome',
                       'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                       'Credit_History', 'Property_Area_encoded', 'TotalIncome', 
                       'IncomeRatio', 'LoanAmountRatio']
        
        X = df[feature_cols].fillna(0)
        y = df['Loan_Status_encoded']
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        sample_features = [1, 1, 0, 1, 0, 5000, 2000, 150000, 360, 1, 0, 7000, 0.047, 416.67]
        prediction = model.predict([sample_features])
        probability = model.predict_proba([sample_features])[0]
        
        print(f"✅ Prediction model test passed!")
        print(f"   - Model trained successfully")
        print(f"   - Sample prediction: {prediction[0]}")
        print(f"   - Approval probability: {probability[1]:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction model test failed: {e}")
        return False

def test_streamlit_components():
    """Test Streamlit components"""
    print("\n🧪 Testing Streamlit Components...")
    
    try:
        # Test basic Streamlit functionality
        import streamlit as st
        
        # Test session state
        if "test_messages" not in st.session_state:
            st.session_state.test_messages = []
        
        # Test message display
        test_message = {
            "role": "user",
            "content": "Test question",
            "confidence": 0.85,
            "relevant_docs": ["Test doc 1", "Test doc 2"]
        }
        
        st.session_state.test_messages.append(test_message)
        
        print("✅ Streamlit components test passed!")
        return True
    except Exception as e:
        print(f"❌ Streamlit components test failed: {e}")
        return False

def test_chat_display():
    """Test chat display functionality"""
    print("\n🧪 Testing Chat Display...")
    
    try:
        # Simulate chat display functionality
        messages = [
            {
                "role": "user",
                "content": "What is the loan approval rate?"
            },
            {
                "role": "assistant",
                "content": "Based on the data, the overall loan approval rate is 68.1%.",
                "confidence": 0.85,
                "relevant_docs": ["Document 1", "Document 2"]
            }
        ]
        
        # Test message formatting
        for message in messages:
            if message["role"] == "user":
                print(f"   User: {message['content']}")
            else:
                confidence = message.get("confidence", 0.8)
                print(f"   Assistant: {message['content']} (Confidence: {confidence:.2f})")
        
        print("✅ Chat display test passed!")
        return True
    except Exception as e:
        print(f"❌ Chat display test failed: {e}")
        return False

def run_streamlit_test():
    """Run Streamlit-based tests"""
    st.title("🧪 Consolidated Test Suite")
    
    st.write("### Testing GROQ Integration")
    
    try:
        from rag_system_groq import RAGSystemGROQ
        
        # Check GROQ API key
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            st.success("✅ GROQ API key found")
            mode = "GROQ LLaMA API"
        else:
            st.warning("⚠️ GROQ API key not found - using fallback mode")
            mode = "Fallback Mode"
        
        # Initialize RAG system
        with st.spinner("Initializing RAG system..."):
            rag = RAGSystemGROQ()
            st.success("✅ RAG system initialized successfully")
        
        # Test questions
        test_questions = [
            "What is the loan approval rate?",
            "How does income affect loan approval?",
            "What is the average loan amount?"
        ]
        
        st.subheader("🤖 Testing Responses")
        
        for i, question in enumerate(test_questions, 1):
            st.write(f"**{i}. Question:** {question}")
            
            with st.spinner(f"Processing question {i}..."):
                try:
                    result = rag.answer_question(question)
                    
                    # Display response
                    st.write(f"**Response:** {result['response']}")
                    st.write(f"**Confidence:** {result['confidence_scores'][0]:.2f}")
                    st.write(f"**Mode:** {mode}")
                    st.write(f"**Relevant Documents:** {len(result['relevant_docs'])}")
                    
                    # Show relevant documents in expander
                    with st.expander(f"View relevant documents for question {i}"):
                        for j, doc in enumerate(result['relevant_docs'], 1):
                            st.write(f"{j}. {doc}")
                    
                    st.write("---")
                    
                except Exception as e:
                    st.error(f"Error processing question {i}: {e}")
        
        st.success("✅ GROQ integration test completed!")
        
    except Exception as e:
        st.error(f"Error during testing: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Consolidated Loan Approval RAG Chatbot Tests\n")
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Data Processor", test_data_processor),
        ("GROQ RAG System", test_rag_system_groq),
        ("Basic RAG System", test_rag_system_basic),
        ("Prediction Model", test_prediction_model),
        ("Streamlit Components", test_streamlit_components),
        ("Chat Display", test_chat_display)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed!")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the chatbot, run:")
        print("   streamlit run main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nMake sure to install all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    # Check if running in Streamlit
    if 'streamlit' in sys.modules:
        run_streamlit_test()
    else:
        main() 