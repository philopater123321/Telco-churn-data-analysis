# Telco Customer Churn prediction model:
This project is made to analyze the data for Telco customers, and create a model that helps them in finding why people are leaving their store 
and provide some business insights and recommendations for them to improve their profit and also to increase the customers who don't leave the store.
### Important note:
As this isn't a website or a game, so GitHub can't deploy it in their servers and provide a demo url instead you can download the exe file which is from this link: [Download the Latest App (.exe)](https://github.com/philopater123321/Telco-churn-data-analysis/releases/latest) just download the zip file and unzip it to be able to open the .exe file easily.
or you can simply download the repository files and put all of them in one folder and you will be able to run the code manually on your computer, and you have the requirements.txt as a file, where you can open and see what are the python libraries you need to install to run the code on your computer.
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

## ROC-AUC score:
This part of the code measures how accurate the model can differentiate between the customers who will churn and who will not.
ROC stands for receiver operating characteristic, and AUC stands for area under the curve.
- 1.0 means best classification
- 0.5 same as random guessing
- <0.5 worse than random guessing
  
My model got 0.84 which is pretty good.

### Business insights:
You can read my docx or word file for my conclusion from the model results and analysis.
## How to run the code:
- #### Place the following in one folder:
- data.csv
- main.py
- install the required libraries in requirements.txt file

