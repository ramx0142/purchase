import streamlit as st
import pickle
import pandas as pd

# 1. Load the trained model
# We use pickle because that is what you used in your notebook
try:
    with open('purchase.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'purchase.pkl' not found. Please download it from your notebook and place it in this folder.")
    st.stop()

# 2. App Title and Description
st.title("🛍️ Purchase Prediction App")
st.write("Enter the customer's Age and Salary to predict if they will buy the product.")

# 3. User Inputs
# We use number_input to ensure we get valid integers
age = st.number_input("Enter Age:", min_value=18, max_value=100, value=25, step=1)
salary = st.number_input("Enter Salary:", min_value=0, value=50000, step=500)

# 4. Prediction Logic
if st.button("Predict Purchase"):
    # In your notebook (Page 3), you created a DataFrame with columns 'age' and 'salary'
    # We must do the same here so the model recognizes the feature names
    user_data = pd.DataFrame({'age': [age], 'salary': [salary]})
    
    try:
        prediction = model.predict(user_data)
        
        # Check result (0 = No Buy, 1 = Buy)
        if prediction[0] == 1:
            st.success("Result: The User **WILL BUY** the product. ✅")
        else:
            st.error("Result: The User **WILL NOT BUY** the product. ❌")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Model: Decision Tree Classifier")
