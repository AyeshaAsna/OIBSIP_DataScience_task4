# OIBSIP_DataScience_task4
OIBSIP Internship – Data Science Task 4:Email spam Detection with Machine Learning
# OIBSIP Machine Learning Task 4: Email Spam Detection

#Objective
This task is part of the Oasis Infobyte Internship Program (OIBSIP).  
The goal of Task 4 is to Email spam Detection with Machine Learning: We’ve all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email
that is sent to a massive number of users at one time, frequently containing cryptic
messages, scams, or most dangerously, phishing content.


In this Project, use Python to build an email spam detector. Then, use machine learning to
train the spam detector to recognize and classify emails into spam and non-spam. Let’s get
started! 
The project demonstrates NLP preprocessing, text vectorization, model training, evaluation, and real-time predictions.

#Steps Performed
1. **Data Loading** – Loaded spam dataset (SMS/email dataset).  
2. **Data Cleaning** – Removed duplicates, missing values, renamed columns, converted labels (spam=1, ham=0).  
3. **Exploratory Data Analysis (EDA)** – Analyzed message lengths, word counts, spam percentages, and visualized distributions.  
4. **Text Preprocessing** – Lowercased text, removed punctuation, numbers, and extra spaces.  
5. **Feature Extraction** – Used **TF-IDF Vectorization** with unigrams and bigrams.  
6. **Model Training** – Trained multiple classifiers:
   - Logistic Regression  
   - Multinomial Naive Bayes  
   - Support Vector Machine (SVM)  
   - Decision Tree  
   - Random Forest  
   - Gradient Boosting  
7. **Model Evaluation** – Compared accuracy, ROC-AUC, confusion matrix, precision, recall, and F1 score.  
8. **Hyperparameter Tuning** – Tuned Random Forest parameters using GridSearchCV.  
9. **Prediction on New Emails** – Classified unseen messages with confidence scores.  
10. **Insights & Recommendations** – Summarized findings and suggested improvements.

# Tools Used
- **Python 3.11.9**  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Dataset:** *spam.csv* (SMS/email dataset)  
- **Editor:** VS Code / Jupyter Notebook  

#Outcome
- Built a complete ML pipeline for spam classification.  
- Achieved **high accuracy (>95%)** with top-performing models (Naive Bayes, Random Forest, Logistic Regression).  
- Generated insights on spam vs ham message length, word count, and feature importance.  
- Created a flexible script for real-time spam detection on new emails.

#Key Visualizations
- Label distribution (spam vs ham).  
- Message length & word count distributions.  
- Box plots of message stats by label.  
- Top 10 most common words in spam messages.  
- Confusion matrix & ROC curve.  
- Feature importance for tree-based models.  

# How to Run
1. Dataset: spam.csv
2. Run the script:
```bash
python task_4.py
