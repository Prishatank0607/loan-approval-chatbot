import pandas as pd
import numpy as np
import requests
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json

class LoanDataProcessor:
    def __init__(self):
        self.dataset_url = "https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction"
        self.data_dir = "data"
        self.processed_data_path = "data/processed_loan_data.csv"
        self.knowledge_base_path = "data/knowledge_base.json"
        
    def download_dataset(self):
        """Download the loan dataset from Kaggle"""
        # Since direct download from Kaggle requires authentication,
        # we'll create a sample dataset based on the description
        print("Creating sample loan dataset based on Kaggle description...")
        
        # Create sample data based on typical loan approval features
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Loan_ID': [f'LP{i:03d}' for i in range(1, n_samples + 1)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Married': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
            'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
            'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
            'ApplicantIncome': np.random.randint(150, 8100, n_samples),
            'CoapplicantIncome': np.random.randint(0, 4167, n_samples),
            'LoanAmount': np.random.randint(9, 700, n_samples),
            'Loan_Amount_Term': np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360], n_samples),
            'Credit_History': np.random.choice([0, 1], n_samples),
            'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples),
            'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Save the dataset
        df.to_csv(self.processed_data_path, index=False)
        print(f"Dataset saved to {self.processed_data_path}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the loan dataset"""
        print("Preprocessing loan dataset...")
        
        # Handle missing values
        df = df.fillna({
            'Gender': df['Gender'].mode()[0],
            'Married': df['Married'].mode()[0],
            'Dependents': df['Dependents'].mode()[0],
            'Self_Employed': df['Self_Employed'].mode()[0],
            'LoanAmount': df['LoanAmount'].median(),
            'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
            'Credit_History': df['Credit_History'].median()
        })
        
        # Create features
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomeRatio'] = df['TotalIncome'] / df['LoanAmount']
        df['LoanAmountRatio'] = df['LoanAmount'] / df['Loan_Amount_Term']
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area', 'Loan_Status']
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Save processed data
        df.to_csv(self.processed_data_path, index=False)
        
        return df, label_encoders
    
    def create_knowledge_base(self, df):
        """Create a knowledge base from the loan dataset"""
        print("Creating knowledge base...")
        
        knowledge_base = {
            "dataset_info": {
                "total_records": len(df),
                "features": list(df.columns),
                "loan_approval_rate": (df['Loan_Status'] == 'Y').mean(),
                "average_loan_amount": df['LoanAmount'].mean(),
                "average_income": df['TotalIncome'].mean()
            },
            "feature_insights": {
                "income_analysis": {
                    "average_applicant_income": df['ApplicantIncome'].mean(),
                    "average_coapplicant_income": df['CoapplicantIncome'].mean(),
                    "income_distribution": df['TotalIncome'].describe().to_dict()
                },
                "loan_analysis": {
                    "loan_amount_stats": df['LoanAmount'].describe().to_dict(),
                    "loan_term_distribution": df['Loan_Amount_Term'].value_counts().to_dict(),
                    "credit_history_distribution": df['Credit_History'].value_counts().to_dict()
                },
                "demographic_analysis": {
                    "gender_distribution": df['Gender'].value_counts().to_dict(),
                    "education_distribution": df['Education'].value_counts().to_dict(),
                    "property_area_distribution": df['Property_Area'].value_counts().to_dict()
                }
            },
            "approval_patterns": {
                "by_gender": df.groupby('Gender')['Loan_Status'].apply(lambda x: (x == 'Y').mean()).to_dict(),
                "by_education": df.groupby('Education')['Loan_Status'].apply(lambda x: (x == 'Y').mean()).to_dict(),
                "by_property_area": df.groupby('Property_Area')['Loan_Status'].apply(lambda x: (x == 'Y').mean()).to_dict(),
                "by_credit_history": df.groupby('Credit_History')['Loan_Status'].apply(lambda x: (x == 'Y').mean()).to_dict()
            },
            "sample_questions": [
                "What is the loan approval rate?",
                "How does income affect loan approval?",
                "What is the average loan amount?",
                "How does credit history impact approval?",
                "Which property area has the highest approval rate?",
                "What is the relationship between education and loan approval?",
                "How does gender affect loan approval rates?",
                "What are the income requirements for loan approval?",
                "What is the typical loan term?",
                "How does having dependents affect loan approval?"
            ]
        }
        
        # Save knowledge base
        with open(self.knowledge_base_path, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"Knowledge base saved to {self.knowledge_base_path}")
        return knowledge_base
    
    def load_data(self):
        """Load or create the dataset and knowledge base"""
        if os.path.exists(self.processed_data_path):
            print("Loading existing processed dataset...")
            df = pd.read_csv(self.processed_data_path)
        else:
            print("Creating new dataset...")
            df = self.download_dataset()
            df, _ = self.preprocess_data(df)
        
        if os.path.exists(self.knowledge_base_path):
            print("Loading existing knowledge base...")
            with open(self.knowledge_base_path, 'r') as f:
                knowledge_base = json.load(f)
        else:
            print("Creating new knowledge base...")
            knowledge_base = self.create_knowledge_base(df)
        
        return df, knowledge_base

if __name__ == "__main__":
    processor = LoanDataProcessor()
    df, knowledge_base = processor.load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Knowledge base keys: {list(knowledge_base.keys())}") 