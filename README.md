# Telco Customer Churn prediction model:
This project is made to analyze the data for Telco customers, and create a model that helps them in finding why people are leaving their store 
and provide some business insights and recommendations for them to improve their profit and also to increase the customers who don't leave the store.

## Libraries used and modules in python:
- pandas for working with the data
- Matplotlib for visualisation
- scikit-learn for preprocessing, training, and evaluation

## The model parts of the code by order:
- Load the data in the code using pandas
- Check the number of the churned customers and don't
- Data clean for total charges and converting them to numerical values and removing rows where it is missing.
- separation of the target column (churn) from the other parameters.
- Split the data into testing and training parts
- preprocessing for numerical and categorial features in sklearn
- Train a Logistic Regresion classifier.
- Getting the accuracy, prediction, recall, F1, confusion matrix, and finally, ROC-AUC to modify the model's accuracy
- See Churn patterns and provide business insights and recommendations.

### The model uses Logistic regression to get the predictions and accuracy:
- test_size = 0.4
- random_state = 42
- stratify = y

## Results:
- before using ROC-AUC:
<img width="570" height="237" alt="image" src="https://github.com/user-attachments/assets/c18e57f2-e8e9-4313-a446-3446012bb58b" />
_________
- after using it:
<img width="395" height="117" alt="image" src="https://github.com/user-attachments/assets/ecddf4bf-963d-46c9-95bd-d9c22442ec4f" />

### Business insights:
You can read my docx or word file for my conclusion from the model results and analysis.
## How to run the code:
- #### Place the following in one folder:
- data.csv
- main.py
- install the required libraries in requirements.txt file

