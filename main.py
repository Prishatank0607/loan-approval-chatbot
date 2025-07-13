import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import LoanDataProcessor
from rag_system_groq import RAGSystemGROQ
from config import Config

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: none;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .user-message {
        background: white;
        color: #333333;
        border-left: 6px solid #4CAF50;
        margin-left: 20%;
        border: 2px solid #e0e0e0;
    }
    
    .bot-message {
        background: #ff69b4;
        color: white;
        border-left: 6px solid #FF9800;
        margin-right: 20%;
    }
    
    /* Confidence bar styling */
    .confidence-bar {
        background-color: rgba(255,255,255,0.3);
        border-radius: 15px;
        height: 12px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 15px;
        transition: width 0.5s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: none;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem 1.2rem;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .stTabs > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    /* Sample questions styling */
    .sample-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 0.8rem 1.5rem;
        margin: 0.5rem 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sample question button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 0.3rem 0;
        width: 100%;
        text-align: left;
        font-size: 14px;
        line-height: 1.4;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Specific styling for sample question buttons */
    .stButton > button:has(span:contains("❓")) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chart styling */
    .plotly-chart {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: white !important;
        background: linear-gradient(135deg, #ff69b4 0%, #f093fb 100%);
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin-top: 5px;
        color: white !important;
    }
    
    .streamlit-expanderContent p {
        color: white !important;
    }
    
    .streamlit-expanderContent strong {
        color: white !important;
    }
    
    /* Better text contrast */
    .stMarkdown {
        color: #333;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Prediction form styling */
    .prediction-form {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .chat-message {
            margin-left: 5% !important;
            margin-right: 5% !important;
        }
    }
    
    /* Chat input container */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 2px solid #e0e0e0;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_rag():
    """Load data and initialize RAG system with caching"""
    try:
        # Initialize data processor
        processor = LoanDataProcessor()
        df, knowledge_base = processor.load_data()
        
        # Initialize RAG system with GROQ
        rag = RAGSystemGROQ()
        
        return df, knowledge_base, rag
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def train_prediction_model(df):
    """Train a loan prediction model"""
    try:
        # Prepare features for prediction
        feature_cols = ['Gender_encoded', 'Married_encoded', 'Dependents_encoded', 
                       'Education_encoded', 'Self_Employed_encoded', 'ApplicantIncome',
                       'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                       'Credit_History', 'Property_Area_encoded', 'TotalIncome', 
                       'IncomeRatio', 'LoanAmountRatio']
        
        X = df[feature_cols].fillna(0)
        y = df['Loan_Status_encoded']
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, feature_cols
    except Exception as e:
        st.error(f"Error training prediction model: {str(e)}")
        return None, None

def create_prediction_interface(df, model, feature_cols):
    """Create the loan prediction interface"""
    st.subheader("🔮 Loan Approval Prediction")
    st.markdown("Enter your details to predict loan approval probability:")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
            married = st.selectbox("Marital Status", ["Yes", "No"], key="pred_married")
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="pred_dependents")
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="pred_education")
            self_employed = st.selectbox("Self Employed", ["Yes", "No"], key="pred_self_employed")
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"], key="pred_property_area")
        
        with col2:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, key="pred_applicant_income")
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=2000, key="pred_coapplicant_income")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=150000, key="pred_loan_amount")
            loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360], key="pred_loan_term")
            credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good" if x == 1 else "Poor", key="pred_credit_history")
    
    # Calculate derived features
    total_income = applicant_income + coapplicant_income
    income_ratio = total_income / loan_amount if loan_amount and loan_amount > 0 else 0
    loan_amount_ratio = loan_amount / loan_term if loan_term and loan_term > 0 else 0
    
    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    dependents_encoded = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents or "0"]
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    property_area_encoded = {"Urban": 0, "Rural": 1, "Semiurban": 2}[property_area or "Urban"]
    
    # Create feature vector
    features = [gender_encoded, married_encoded, dependents_encoded, education_encoded,
                self_employed_encoded, applicant_income, coapplicant_income, loan_amount,
                loan_term, credit_history, property_area_encoded, total_income, income_ratio, loan_amount_ratio]
    
    if st.button("🔮 Predict Loan Approval", key="predict_button"):
        if model is not None:
            try:
                # Make prediction
                prediction_proba = model.predict_proba([features])[0]
                approval_probability = prediction_proba[1]  # Probability of approval
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Approval Probability", f"{approval_probability:.1%}")
                
                with col2:
                    status = "✅ Approved" if approval_probability > 0.5 else "❌ Rejected"
                    st.metric("Prediction", status)
                
                with col3:
                    confidence = max(approval_probability, 1 - approval_probability)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Create prediction visualization
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=approval_probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Loan Approval Probability"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 50], 'color': "orange"},
                            {'range': [50, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Provide recommendations
                st.markdown("### 💡 Recommendations")
                if approval_probability > 0.7:
                    st.success("🎉 High approval probability! Your application looks strong.")
                elif approval_probability > 0.5:
                    st.warning("⚠️ Moderate approval probability. Consider improving your application.")
                else:
                    st.error("❌ Low approval probability. Consider the following improvements:")
                    recommendations = []
                    if income_ratio < 5:
                        recommendations.append("Increase your income or reduce loan amount")
                    if credit_history == 0:
                        recommendations.append("Improve your credit history")
                    if loan_amount_ratio > 1000:
                        recommendations.append("Consider a longer loan term")
                    if total_income < 5000:
                        recommendations.append("Increase your total income")
                    
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.error("Prediction model not available. Please try again later.")

def create_chat_interface(rag, knowledge_base, df):
    """Create the chat interface with improved interactivity"""
    st.subheader("💬 Ask Questions About Loan Approval")
    
    # Initialize chat history and input state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    # Display sample questions
    st.markdown("**💡 Sample Questions:**")
    sample_questions = knowledge_base['sample_questions'][:5]
    
    # Create buttons in a more compact layout with better styling
    for i, question in enumerate(sample_questions):
        if st.button(f"❓ {question}", key=f"sample_{i}"):
            if st.session_state.get("last_processed_input") != question:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get response from RAG system
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = rag.answer_question(question)
                        
                        # Check if result has the expected structure
                        if not isinstance(result, dict):
                            st.error(f"Unexpected result type: {type(result)}")
                            raise ValueError(f"Expected dict, got {type(result)}")
                        
                        if 'response' not in result:
                            st.error("Result missing 'response' key")
                            raise ValueError("Result missing 'response' key")
                        
                        # Handle confidence scores safely
                        confidence_scores = result.get('confidence_scores', [0.8])
                        if isinstance(confidence_scores, (list, tuple)) and len(confidence_scores) > 0:
                            confidence = float(confidence_scores[0])
                        else:
                            confidence = 0.8

                        # Add bot response to chat
                        assistant_message = {
                            "role": "assistant", 
                            "content": result['response'],
                            "confidence": confidence,
                            "relevant_docs": result.get('relevant_docs', []),
                            "query": question
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # Store the processed input and rerun
                        st.session_state.last_processed_input = question
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "Sorry, I encountered an error while processing your question. Please try again.",
                            "confidence": 0.5,
                            "relevant_docs": []
                        })
                        st.rerun()
    
    # Chat input with improved interactivity
    st.markdown("### 💬 Chat")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            try:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message" style="text-align: right;">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence = message.get("confidence", 0.8)
                    confidence_percent = confidence * 100
                    
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong> {message["content"]}
                        <div style="margin-top: 0.5rem;">
                            <small>Confidence: {confidence_percent:.1f}%</small>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show relevant documents in expander
                    if "relevant_docs" in message and message["relevant_docs"]:
                        with st.expander("📄 View Relevant Information"):
                            for j, doc in enumerate(message["relevant_docs"], 1):
                                st.markdown(f"**{j}.** {doc}")
            except Exception as e:
                st.error(f"Error displaying message {i}: {str(e)}")
    
    # Chat input at the bottom
    st.markdown("---")
    
    # Create input area with send button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key=f"user_input_{st.session_state.input_key}",
            placeholder="Ask about loan approval, requirements, or any questions...",
            on_change=None
        )
    
    with col2:
        send_button = st.button("➤ Send", key="send_button")
    
    # Handle enter key and send button
    if (user_input and user_input.strip()) or send_button:
        input_text = user_input.strip() if user_input else ""
        
        # Check if this is a new input (not already processed)
        if input_text and input_text != st.session_state.get("last_processed_input", ""):
            # Store the input to prevent reprocessing
            st.session_state.last_processed_input = input_text
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": input_text})
            
            # Get response from RAG system
            with st.spinner("🤔 Thinking..."):
                try:
                    result = rag.answer_question(input_text)
                    
                    # Check if result has the expected structure
                    if not isinstance(result, dict):
                        st.error(f"Unexpected result type: {type(result)}")
                        raise ValueError(f"Expected dict, got {type(result)}")
                    
                    if 'response' not in result:
                        st.error("Result missing 'response' key")
                        raise ValueError("Result missing 'response' key")
                    
                    # Handle confidence scores safely
                    confidence_scores = result.get('confidence_scores', [0.8])
                    if isinstance(confidence_scores, (list, tuple)) and len(confidence_scores) > 0:
                        confidence = float(confidence_scores[0])
                    else:
                        confidence = 0.8

                    # Add bot response to chat
                    assistant_message = {
                        "role": "assistant", 
                        "content": result['response'],
                        "confidence": confidence,
                        "relevant_docs": result.get('relevant_docs', []),
                        "query": input_text
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Clear the input by incrementing the key and rerun
                    st.session_state.input_key += 1
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I encountered an error while processing your question. Please try again.",
                        "confidence": 0.5,
                        "relevant_docs": []
                    })

    
    # Clear chat button
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.last_processed_input = ""
        st.session_state.input_key += 1
        st.rerun()

def create_dashboard(df, knowledge_base):
    """Create dashboard with data visualizations"""
    st.subheader("📊 Loan Approval Dashboard")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{knowledge_base['dataset_info']['total_records']:,}")
    
    with col2:
        approval_rate = knowledge_base['dataset_info']['loan_approval_rate']
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    
    with col3:
        avg_loan = knowledge_base['dataset_info']['average_loan_amount']
        st.metric("Avg Loan Amount", f"${avg_loan:,.0f}")
    
    with col4:
        avg_income = knowledge_base['dataset_info']['average_income']
        st.metric("Avg Income", f"${avg_income:,.0f}")
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval by Gender
        gender_rates = knowledge_base['approval_patterns']['by_gender']
        fig_gender = px.bar(
            x=list(gender_rates.keys()),
            y=list(gender_rates.values()),
            title="Approval Rate by Gender",
            labels={'x': 'Gender', 'y': 'Approval Rate'},
            color=list(gender_rates.values()),
            color_continuous_scale='Blues'
        )
        fig_gender.update_layout(showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Approval by Education
        edu_rates = knowledge_base['approval_patterns']['by_education']
        fig_edu = px.bar(
            x=list(edu_rates.keys()),
            y=list(edu_rates.values()),
            title="Approval Rate by Education",
            labels={'x': 'Education', 'y': 'Approval Rate'},
            color=list(edu_rates.values()),
            color_continuous_scale='Greens'
        )
        fig_edu.update_layout(showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)
    
    # Property Area Analysis
    area_rates = knowledge_base['approval_patterns']['by_property_area']
    fig_area = px.pie(
        values=list(area_rates.values()),
        names=list(area_rates.keys()),
        title="Approval Rate by Property Area"
    )
    st.plotly_chart(fig_area, use_container_width=True)

def create_relevant_visualization(query, knowledge_base, df):
    """Create relevant visualizations based on the query"""
    query_lower = query.lower()
    
    # Loan approval rate visualization
    if any(word in query_lower for word in ['approval rate', 'approval', 'rate']):
        approval_rate = knowledge_base['dataset_info']['loan_approval_rate']
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=approval_rate * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Loan Approval Rate"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        return fig, "📊 Loan Approval Rate Gauge"
    
    # Average loan amount visualization
    elif any(word in query_lower for word in ['average loan', 'loan amount', 'amount']):
        avg_loan = knowledge_base['dataset_info']['average_loan_amount']
        loan_stats = knowledge_base['feature_insights']['loan_analysis']['loan_amount_stats']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Min', 'Average', 'Max'],
            y=[loan_stats['min'], avg_loan, loan_stats['max']],
            marker_color=['#ff7f0e', '#2ca02c', '#d62728']
        ))
        fig.update_layout(
            title="Loan Amount Distribution",
            xaxis_title="Loan Amount Categories",
            yaxis_title="Amount ($)",
            height=400
        )
        return fig, "📈 Loan Amount Statistics"
    
    # Income analysis visualization
    elif any(word in query_lower for word in ['income', 'salary', 'earnings']):
        income_info = knowledge_base['feature_insights']['income_analysis']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Applicant Income', 'Coapplicant Income'],
            y=[income_info['average_applicant_income'], income_info['average_coapplicant_income']],
            marker_color=['#1f77b4', '#ff7f0e']
        ))
        fig.update_layout(
            title="Average Income Comparison",
            xaxis_title="Income Type",
            yaxis_title="Amount ($)",
            height=400
        )
        return fig, "💰 Income Analysis"
    
    # Credit history visualization
    elif any(word in query_lower for word in ['credit', 'credit history', 'credit score']):
        credit_dist = knowledge_base['feature_insights']['loan_analysis']['credit_history_distribution']
        
        if isinstance(credit_dist, dict) and len(credit_dist) >= 2:
            keys = list(credit_dist.keys())
            values = list(credit_dist.values())
            
            fig = go.Figure(data=[go.Pie(labels=keys, values=values)])
            fig.update_layout(
                title="Credit History Distribution",
                height=400
            )
            return fig, "💳 Credit History Distribution"
    
    # Gender-based analysis
    elif any(word in query_lower for word in ['gender', 'male', 'female']):
        gender_rates = knowledge_base['approval_patterns']['by_gender']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(gender_rates.keys()),
            y=list(gender_rates.values()),
            marker_color=['#ff9999', '#66b3ff']
        ))
        fig.update_layout(
            title="Approval Rate by Gender",
            xaxis_title="Gender",
            yaxis_title="Approval Rate (%)",
            height=400
        )
        return fig, "👥 Gender-based Approval Rates"
    
    # Education-based analysis
    elif any(word in query_lower for word in ['education', 'graduate', 'not graduate']):
        edu_rates = knowledge_base['approval_patterns']['by_education']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(edu_rates.keys()),
            y=list(edu_rates.values()),
            marker_color=['#ffcc99', '#99ccff']
        ))
        fig.update_layout(
            title="Approval Rate by Education",
            xaxis_title="Education Level",
            yaxis_title="Approval Rate (%)",
            height=400
        )
        return fig, "🎓 Education-based Approval Rates"
    
    # Property area analysis
    elif any(word in query_lower for word in ['property', 'area', 'urban', 'rural', 'semiurban']):
        area_rates = knowledge_base['approval_patterns']['by_property_area']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(area_rates.keys()),
            y=list(area_rates.values()),
            marker_color=['#ff9999', '#66b3ff', '#99ff99']
        ))
        fig.update_layout(
            title="Approval Rate by Property Area",
            xaxis_title="Property Area",
            yaxis_title="Approval Rate (%)",
            height=400
        )
        return fig, "🏠 Property Area Analysis"
    
    # Default visualization for general queries
    else:
        # Show overall statistics
        approval_rate = knowledge_base['dataset_info']['loan_approval_rate']
        avg_loan = knowledge_base['dataset_info']['average_loan_amount']
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=approval_rate * 100,
            title={'text': "Overall Approval Rate (%)"},
            delta={'reference': 50}
        ))
        fig.update_layout(height=200)
        return fig, "📊 Overall Statistics"
    
    return None, None

def create_sidebar(df, knowledge_base):
    """Create sidebar with additional information"""
    st.sidebar.markdown('<div class="sidebar-header">📈 Dataset Insights</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    st.sidebar.markdown("### 📊 Key Statistics")
    
    # Income analysis
    income_info = knowledge_base['feature_insights']['income_analysis']
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Average Applicant Income:</strong><br>
        ${income_info['average_applicant_income']:,.0f}
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Average Coapplicant Income:</strong><br>
        ${income_info['average_coapplicant_income']:,.0f}
    </div>
    """, unsafe_allow_html=True)
    
    # Loan analysis
    loan_info = knowledge_base['feature_insights']['loan_analysis']
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Loan Amount Range:</strong><br>
        ${loan_info['loan_amount_stats']['min']:,.0f} - ${loan_info['loan_amount_stats']['max']:,.0f}
    </div>
    """, unsafe_allow_html=True)
    
    # Credit history
    credit_dist = loan_info['credit_history_distribution']
    # Handle the case where credit history keys might be strings or missing
    if isinstance(credit_dist, dict) and len(credit_dist) >= 2:
        keys = list(credit_dist.keys())
        if len(keys) >= 2:
            good_credit_rate = credit_dist[keys[1]] / (credit_dist[keys[0]] + credit_dist[keys[1]]) if (credit_dist[keys[0]] + credit_dist[keys[1]]) > 0 else 0
        else:
            good_credit_rate = 0.0
    else:
        good_credit_rate = 0.0
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Good Credit History Rate:</strong><br>
        {good_credit_rate:.1%}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Approval Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Q&A System powered by LLaMA via GROQ API")
    
    # Load data and RAG system
    with st.spinner("🔄 Loading data and initializing RAG system..."):
        df, knowledge_base, rag = load_data_and_rag()
    
    if df is None or knowledge_base is None or rag is None:
        st.error("❌ Failed to load data. Please check your setup and try again.")
        return
    
    # Train prediction model
    with st.spinner("🤖 Training prediction model..."):
        model, feature_cols = train_prediction_model(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔮 Prediction", "📊 Dashboard", "ℹ️ About"])
    
    with tab1:
        create_chat_interface(rag, knowledge_base, df)
    
    with tab2:
        create_prediction_interface(df, model, feature_cols)
    
    with tab3:
        create_dashboard(df, knowledge_base)
    
    with tab4:
        st.markdown("""
        ## About This RAG Chatbot with GROQ
        
        This is a **Retrieval-Augmented Generation (RAG)** chatbot specifically designed for loan approval analysis. 
        It combines the power of document retrieval with GROQ's LLaMA models to provide intelligent responses about loan approval patterns.
        
        ### 🔧 How It Works
        
        1. **Document Retrieval**: The system uses semantic search to find relevant information from the loan dataset
        2. **Context Generation**: Retrieved documents are used as context for generating responses
        3. **Intelligent Response**: A lightweight language model generates human-like responses based on the retrieved context
        
        ### 📊 Dataset Features
        
        - **Loan_ID**: Unique identifier for each loan application
        - **Demographics**: Gender, marital status, dependents, education
        - **Financial**: Applicant income, coapplicant income, loan amount
        - **Loan Details**: Loan term, credit history, property area
        - **Target**: Loan approval status (Y/N)
        
        ### 🛠️ Technology Stack
        
        - **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
        - **Vector Database**: FAISS for efficient similarity search
        - **Generation Model**: GROQ API with LLaMA-3-8B-8192 for response generation
        - **UI Framework**: Streamlit for interactive web interface
        - **Visualization**: Plotly for dynamic charts and graphs
        
        ### 🎯 Use Cases
        
        - Analyze loan approval patterns
        - Understand factors affecting approval rates
        - Explore demographic and financial insights
        - Get data-driven answers about loan approval criteria
        """)
    
    # Create sidebar
    create_sidebar(df, knowledge_base)

if __name__ == "__main__":
    main()
