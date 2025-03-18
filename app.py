import streamlit as st
import shap 
import pandas p
import xgboost as xgb
st.write("""
# Fraud Transaction Detection Website

Hi team, 

This website is to help you to detect illegal transaction from customers and prevent any other possible worse outcomes. 
Please input information of customer demographics and transaction details in the parameter on the left hand side.
We hope you guys enjoy using the app. 


Thank you so much from our team. 
""")
st.write('---')
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Input Parameters')
import streamlit as st
import pandas as pd

def user_input_features():
    st.sidebar.header("User Input ")

    # Numerical Inputs
    Age = st.sidebar.slider('Age', min_value=0, max_value=100, value=10)
    NumDependents = st.sidebar.slider('Number of Dependents', min_value=0, max_value=10, value=1)
    UserTenure = st.sidebar.slider('User Tenure (Months)', min_value=0, max_value=240, value=12)
    Income = st.sidebar.slider('User Income', min_value=0, max_value=1000000, value=20000)
    Expenditure = st.sidebar.slider('Expenditure', min_value=0, max_value=1000000, value=20000)

    # Categorical Inputs (Using selectbox)
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    Occupation = st.sidebar.selectbox('Occupation', ['Student', 'Professional', 'Self-Employed', 'Retired'])
    EducationLevel = st.sidebar.selectbox('Education Level', ['High School', 'Bachelors', 'Masters', 'PhD'])
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
    TransactionType = st.sidebar.selectbox('Transaction Type', ['Online', 'In-store'])
    DeviceType = st.sidebar.selectbox('Device Type', ['Mobile', 'Desktop', 'Tablet'])
    # Transaction Amount 
    TransactionAmount = st.sidebar.number_input('Transaction Amount ($)', min_value=0.0, max_value=10000.0, value=50.0)

    # Transaction Date 
    TransactionDate = st.sidebar.date_input('Transaction Date')

    # Terrorism Flag 
    Terrorism = st.sidebar.checkbox('Terrorism Or Not')

    # Latitude & Longitude (Handling missing values)
    Latitude = st.sidebar.number_input('Latitude', value=0.0, format="%.6f")
    Longitude = st.sidebar.number_input('Longitude', value=0.0, format="%.6f")

    # Create DataFrame
    data = {
        'Age': Age,
        'NumDependents': NumDependents,
        'UserTenure': UserTenure,
        'Gender': Gender,
        'Occupation': Occupation,
        'EducationLevel': EducationLevel,
        'MaritalStatus': MaritalStatus,
        'Income': Income,
        'Expenditure': Expenditure,
        'TransactionType': TransactionType,
        'DeviceType': DeviceType,
        'TransactionAmount': TransactionAmount,
        'TransactionDate': str(TransactionDate),
        'Terrorism': int(Terrorism),
        'Latitude': Latitude,
        'Longitude': Longitude
    }

    # Convert to DataFrame
    return pd.DataFrame(data, index=[0])
    


# Run function and display user inputs
input_features = user_input_features()
st.write("User Input Features:", input_features)
# Build Regression Model
df = input_features()
def shorten_age(age):
    if abs(age) >= 1000:
        return int(str(age)[:2])  
    else:
        return age  

def positive_age(age):
    if age < 0: 
        return -age
    else:
        return age

df['Age'] = df['Age'].apply(shorten_age).apply(positive_age)
df['DeviceType'] = df['DeviceType'].replace({'mob':'Mobile', 'iphone 15' : 'Mobile', 'android': 'Mobile', 'smartphone': 'Mobile', 'galaxys7': 'Mobile'}) 
df['Gender'] = df['Gender'].replace({'he':'Male', 'man' : 'Male', 'isnotfemale': 'Male'})
df['Gender'] = df['Gender'].replace({'fem':'Female', 'isnotmale' : 'Female', 'woman': 'Female', 'she': 'Female'})
df['Gender'] = df['Gender'].replace('Male', '0')
df['Gender'] = df['Gender'].replace('Female', '1')
df['Income'] = df['Income'].str.replace('$','').str.replace( 'AU$' , '').str.replace('AUD' ,'').str.replace('AU', '').astype(float)
# Convert Expenditure:
df['Expenditure'] = df['Expenditure'].replace({'AU$36604.93': '36604.93'}, regex=True)
# Clean the 'Expenditure' column by removing currency symbols and unwanted characters
df['Expenditure'] = df['Expenditure'].replace({'AU\$': '', 'AUD': '', 'AED': '', ' ': ''}, regex=True)

# Convert the cleaned column to numeric values
df['Expenditure'] = pd.to_numeric(df['Expenditure'], errors='coerce')
def change_currency(n):
    n = str(n)
    
    if n.endswith('AUD') or n.startswith('AU$'):
        n = n.replace('AU$', '').replace('AUD', '').strip()
        try:
            n = float(n)
            return n / 1.96  
        except ValueError:
            return None  
    else:
        n = n.replace('GBP', '').replace('£', '').replace('¬', '').strip()
        try:
            n = float(n)
            return n
        except ValueError:
            return None
df['GiftsTransaction'] = df['GiftsTransaction'].apply(change_currency)
df['TransactionAmount'] = df['TransactionAmount'].replace({'AU\$': '', 'AUD': '', 'AED': '', ' ': ''}, regex=True)
df = df.drop('TransactionNumber', axis = 1)
df = df.drop('UserID', axis = 1)
df['TransactionAmount'] = pd.to_numeric(df['TransactionAmount'], errors='coerce').astype('float64')
df = df.drop('EmailDomain', axis = 1)
df = df.drop('Latitude', axis = 1)
df = df.drop('Longitude', axis = 1)
# Get all the categorical columns:
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('TransactionDate')
categorical_columns = categorical_columns.drop('TransactionTime')
categorical_columns = categorical_columns.drop('TransactionLocation')
categorical_columns = categorical_columns.drop('MerchantID')
encode1 = LabelEncoder()
df['TransactionLocation'] = encode1.fit_transform(df['TransactionLocation'])
df['Terrorism'] = encode1.fit_transform(df[['Terrorism']])
df['TransactionDate'] = encode1.fit_transform(df['TransactionDate'])
df['TransactionTime'] = encode1.fit_transform(df['TransactionTime'])
df['MerchantID'] = encode1.fit_transform(df['MerchantID'])
from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df, columns=categorical_columns, dtype = int)
X = df
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler ()
X_train_normalized = scaler.fit_transform(X_train)
model = XGBoost()
xgb_clf = xgb.XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=9,
    min_child_weight=1,
    n_estimators=300,
    subsample=1.0)
prediction = xgb_clf.predict(X_train_normalized)
st.header('Prediction of this transaction')
st.write(prediction)
st.write('---')


