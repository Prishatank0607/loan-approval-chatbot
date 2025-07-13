# 🏦 Loan Approval Chatbot

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot designed for intelligent loan approval analysis. This system combines document retrieval with generative AI to provide data-driven insights about loan approval patterns and personalized loan prediction capabilities.

## About the Project

This project implements a comprehensive loan approval analysis system that combines:

- **RAG (Retrieval-Augmented Generation)** for intelligent question answering
- **Machine Learning** for loan approval prediction
- **Interactive Chat Interface** with confidence scoring
- **Data Visualization** for insights and patterns
- **GROQ API Integration** with LLaMA models for natural language processing

The system provides both analytical insights about loan approval patterns and personalized prediction capabilities, making it a complete solution for loan approval analysis.

## Dataset Overview

### Dataset Features
The system works with comprehensive loan approval data including:

- **Demographics**: Gender, marital status, dependents, education
- **Financial**: Applicant income, coapplicant income, loan amount
- **Loan Details**: Loan term, credit history, property area
- **Target**: Loan approval status (Y/N)

### Dataset Sources
- **Primary Dataset**: [Loan Approval Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
### Dataset Statistics
- **Total Records**: 614 loan applications
- **Approval Rate**: ~69%
- **Features**: 12 input features + 1 target variable
- **Data Quality**: Clean, preprocessed, and feature-engineered

## 🧠 Approach

### 1. **RAG Architecture**
```
User Query → Document Retrieval → Context Generation → Response Generation
```

- **Document Retrieval**: Uses FAISS vector database with sentence transformers
- **Context Generation**: Retrieved documents provide relevant context
- **Response Generation**: GROQ LLaMA models generate human-like answers

### 2. **Machine Learning Pipeline**
```
Data Preprocessing → Feature Engineering → Model Training → Prediction
```

- **Feature Engineering**: Creates derived features from user input
- **Model Training**: Random Forest classifier with cross-validation
- **Prediction**: Provides approval probability and recommendations

### 3. **Multi-Modal Interface**
- **Chat Interface**: Natural language Q&A with confidence scoring
- **Prediction Interface**: Interactive form for loan approval prediction
- **Dashboard**: Data visualizations and statistical insights

## Features Implemented

### **Intelligent Chat Interface**
- Natural language question processing
- Confidence scoring for responses
- Relevant document highlighting
- Chat history with clear message distinction
- Enter key support for easy messaging
- Greeting and general conversation support

### **Loan Prediction System**
- Interactive form for loan details input
- Real-time approval probability calculation
- Visual gauge showing approval likelihood
- Personalized improvement recommendations
- Confidence metrics and status prediction

### **Interactive Dashboard**
- Key metrics display
- Interactive visualizations
- Approval rate analysis by demographics
- Property area analysis
- Income distribution charts

### **Modern UI/UX**
- Responsive design with gradient backgrounds
- Color-coded chat messages (white for user, pink for bot)
- Professional styling with hover effects
- Sample questions for quick start
- Sidebar with dataset insights

### **Technical Features**
- Environment variable configuration (.env support)
- Error handling and fallback responses
- Session state management
- Caching for performance optimization
- Comprehensive logging and debugging

##  Tools and Technologies

### **Core Technologies**
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **GROQ API**: LLaMA model integration
- **FAISS**: Vector database for similarity search

### **Machine Learning & AI**
- **Sentence Transformers**: Text embedding model (all-MiniLM-L6-v2)
- **Scikit-learn**: Random Forest classifier
- **NumPy & Pandas**: Data manipulation
- **LLaMA-3-8B-8192**: Language model via GROQ

### **Data Visualization**
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical plotting

### **Development & Deployment**
- **Git**: Version control
- **Streamlit Cloud**: Deployment platform

## 🚀 How to Run

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- GROQ API key (free from [console.groq.com](https://console.groq.com/))

### **Installation Steps**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-approval-chatbot.git
   cd loan-approval-chatbot
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   
   Get your free GROQ API key from: [https://console.groq.com/](https://console.groq.com/)

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

### **Quick Start Commands**
```bash
# One-liner setup (if you have Python and pip)
git clone https://github.com/yourusername/loan-approval-chatbot.git && \
cd loan-approval-chatbot && \
pip install -r requirements.txt && \
echo "GROQ_API_KEY=your_key_here" > .env && \
streamlit run main.py
```

## Deployment

### **Streamlit Cloud Deployment**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub repository
   - Set environment variables:
     - `GROQ_API_KEY`: Your GROQ API key
   - Deploy!


#### **Local Network Access**
```bash
# Run on local network
streamlit run main.py --server.address=0.0.0.0 --server.port=8501
```

## Useful Links

- **GROQ API**: [https://console.groq.com/](https://console.groq.com/)
- **Streamlit Cloud**: [https://share.streamlit.io/](https://share.streamlit.io/)
- **Dataset Source**: [https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **FAISS Documentation**: [https://faiss.ai/](https://faiss.ai/)
- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
---

**Happy Chatting! 🚀**

*Built with ❤️ using RAG and AI*
