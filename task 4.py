# Email Spam Detection with Machine Learning
# This script builds an email spam detector using ML to classify emails into spam and non-spam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and examine the spam dataset"""
    print("Loading Spam Dataset...")
    
    try:
        # Load the CSV file
        df = pd.read_csv('datasets/spam.csv', encoding='latin-1')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        return df
    
    except FileNotFoundError:
        print("Error: File 'datasets/spam.csv' not found!")
        print("Please make sure the file is in the correct location.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the spam dataset"""
    print("\n" + "="*60)
    print("DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Check for missing values
    print("Missing values in each column:")
    print(df_clean.isnull().sum())
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    print(f"\nShape after removing missing values: {df_clean.shape}")
    
    # Clean column names (remove extra spaces and special characters)
    df_clean.columns = df_clean.columns.str.strip()
    
    # Check for duplicate rows
    duplicates = df_clean.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Shape after removing duplicates: {df_clean.shape}")
    
    # Rename columns for clarity
    df_clean.columns = ['label', 'message'] + [f'col_{i}' for i in range(2, len(df_clean.columns))]
    
    # Keep only relevant columns
    df_clean = df_clean[['label', 'message']]
    
    # Convert labels to binary (spam = 1, ham = 0)
    df_clean['label'] = df_clean['label'].map({'spam': 1, 'ham': 0})
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Label distribution:")
    print(df_clean['label'].value_counts())
    print(f"Spam percentage: {(df_clean['label'].mean() * 100):.2f}%")
    
    return df_clean

def text_preprocessing(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove numbers (optional - you can keep them if they're important)
    text = re.sub(r'\d+', '', text)
    
    return text

def explore_data(df):
    """Explore the dataset with basic statistics and visualizations"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic statistics
    print("Basic Statistics:")
    print(df.describe())
    
    # Text length analysis
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    
    print(f"\nMessage length statistics:")
    print(f"Average message length: {df['message_length'].mean():.2f} characters")
    print(f"Average word count: {df['word_count'].mean():.2f} words")
    
    # Create visualizations
    create_exploratory_plots(df)

def create_exploratory_plots(df):
    """Create various exploratory plots for spam data"""
    print("Creating exploratory visualizations...")
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Label Distribution
    plt.subplot(3, 3, 1)
    label_counts = df['label'].value_counts()
    colors = ['lightgreen', 'lightcoral']
    bars = plt.bar(['Ham (0)', 'Spam (1)'], label_counts.values, color=colors)
    plt.xlabel('Email Type')
    plt.ylabel('Count')
    plt.title('Distribution of Email Types')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Message Length Distribution
    plt.subplot(3, 3, 2)
    plt.hist(df['message_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Message Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Message Lengths')
    plt.grid(True, alpha=0.3)
    
    # 3. Word Count Distribution
    plt.subplot(3, 3, 3)
    plt.hist(df['word_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Counts')
    plt.grid(True, alpha=0.3)
    
    # 4. Message Length by Label
    plt.subplot(3, 3, 4)
    df.boxplot(column='message_length', by='label', ax=plt.gca())
    plt.title('Message Length by Email Type')
    plt.suptitle('')  # Remove default suptitle
    plt.xlabel('Email Type (0=Ham, 1=Spam)')
    plt.grid(True, alpha=0.3)
    
    # 5. Word Count by Label
    plt.subplot(3, 3, 5)
    df.boxplot(column='word_count', by='label', ax=plt.gca())
    plt.title('Word Count by Email Type')
    plt.suptitle('')  # Remove default suptitle
    plt.xlabel('Email Type (0=Ham, 1=Spam)')
    plt.grid(True, alpha=0.3)
    
    # 6. Scatter Plot: Length vs Word Count
    plt.subplot(3, 3, 6)
    colors = ['green' if label == 0 else 'red' for label in df['label']]
    plt.scatter(df['word_count'], df['message_length'], c=colors, alpha=0.6)
    plt.xlabel('Word Count')
    plt.ylabel('Message Length (characters)')
    plt.title('Message Length vs Word Count')
    plt.legend(['Ham', 'Spam'], loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 7. Average Length by Label
    plt.subplot(3, 3, 7)
    avg_length_by_label = df.groupby('label')['message_length'].mean()
    bars = plt.bar(['Ham', 'Spam'], avg_length_by_label.values, color=['lightgreen', 'lightcoral'])
    plt.xlabel('Email Type')
    plt.ylabel('Average Message Length')
    plt.title('Average Message Length by Email Type')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 8. Average Word Count by Label
    plt.subplot(3, 3, 8)
    avg_words_by_label = df.groupby('label')['word_count'].mean()
    bars = plt.bar(['Ham', 'Spam'], avg_words_by_label.values, color=['lightgreen', 'lightcoral'])
    plt.xlabel('Email Type')
    plt.ylabel('Average Word Count')
    plt.title('Average Word Count by Email Type')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 9. Common Words Analysis
    plt.subplot(3, 3, 9)
    # Get all words from messages
    all_words = ' '.join(df['message']).lower().split()
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
    filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    top_words = dict(word_freq.most_common(10))
    
    bars = plt.bar(range(len(top_words)), list(top_words.values()), color='gold')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words')
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    """Prepare data for machine learning"""
    print("\n" + "="*60)
    print("DATA PREPARATION FOR ML")
    print("="*60)
    
    # Apply text preprocessing
    print("Applying text preprocessing...")
    df['cleaned_message'] = df['message'].apply(text_preprocessing)
    
    # Split data into features and target
    X = df['cleaned_message']
    y = df['label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Feature extraction using TF-IDF
    print("\nExtracting features using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Feature matrix shape after TF-IDF: {X_train_tfidf.shape}")
    print(f"Number of features: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # ROC-AUC score if probabilities are available
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")
    
    return results

def evaluate_models(results, y_test):
    """Evaluate and compare model performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Compare accuracies
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    
    plt.figure(figsize=(15, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    names = list(accuracies.keys())
    values = list(accuracies.values())
    bars = plt.bar(names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Cross-validation comparison
    plt.subplot(1, 2, 2)
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    
    bars = plt.bar(names, cv_means, yerr=cv_stds, capsize=5, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('Cross-validation Score')
    plt.title('Cross-validation Performance')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, cv_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = results[best_model_name]['model']
    best_accuracy = accuracies[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return best_model_name, best_model

def detailed_evaluation(best_model, best_model_name, X_test, y_test, vectorizer):
    """Detailed evaluation of the best model"""
    print("\n" + "="*60)
    print(f"DETAILED EVALUATION: {best_model_name}")
    print("="*60)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    
    # 2. ROC Curve
    if y_pred_proba is not None:
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top 20 features
        top_indices = np.argsort(feature_importance)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        plt.subplot(2, 3, 3)
        bars = plt.barh(range(len(top_features)), top_importance, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top 20 Feature Importance - {best_model_name}')
    
    # 4. Prediction distribution
    plt.subplot(2, 3, 4)
    if y_pred_proba is not None:
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Ham', color='lightgreen')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Spam', color='lightcoral')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Model performance by message length
    plt.subplot(2, 3, 5)
    # This would require the original test data with message lengths
    # For now, we'll show a placeholder
    plt.text(0.5, 0.5, 'Message Length Analysis\n(Would require original test data)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Performance by Message Length')
    plt.axis('off')
    
    # 6. Model comparison summary
    plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    # Calculate precision, recall, F1 from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    values = [accuracy, precision, recall, f1]
    bars = plt.bar(metrics, values, color=['gold', 'lightblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for the best model"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Tune Random Forest (usually performs well on text data)
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Tuning Random Forest hyperparameters...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def make_predictions(best_model, vectorizer):
    """Make predictions on new email data"""
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Example new emails (you can modify these values)
    new_emails = [
        "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
        "Hi John, can you please send me the meeting notes from yesterday? Thanks!",
        "URGENT: Your account has been suspended. Click here to verify your identity immediately!",
        "The project deadline has been extended to next Friday. Please update your schedule accordingly.",
        "FREE VIAGRA NOW! Limited time offer. Don't miss out on this amazing deal!",
        "Meeting reminder: Team standup at 9 AM tomorrow. Please prepare your updates.",
        "You've been selected for a special offer! 50% off all products. Act now!",
        "Hi Sarah, I've attached the quarterly report. Please review and let me know if you have any questions."
    ]
    
    print("Predictions on new email data:")
    for i, email in enumerate(new_emails, 1):
        # Preprocess the email
        cleaned_email = text_preprocessing(email)
        
        # Transform using the vectorizer
        email_features = vectorizer.transform([cleaned_email])
        
        # Make prediction
        prediction = best_model.predict(email_features)[0]
        prediction_proba = best_model.predict_proba(email_features)[0]
        
        # Get confidence
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        print(f"\nEmail {i}:")
        print(f"Text: {email[:100]}{'...' if len(email) > 100 else ''}")
        print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Spam Probability: {prediction_proba[1]:.3f}")
        print(f"Ham Probability: {prediction_proba[0]:.3f}")

def generate_insights(df, results):
    """Generate insights and recommendations"""
    print("\n" + "="*60)
    print("INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # 1. Overall dataset insights
    total_emails = len(df)
    spam_count = df['label'].sum()
    ham_count = total_emails - spam_count
    spam_percentage = (spam_count / total_emails) * 100
    
    insights.append(f"Total emails analyzed: {total_emails}")
    insights.append(f"Spam emails: {spam_count} ({spam_percentage:.1f}%)")
    insights.append(f"Ham emails: {ham_count} ({100 - spam_percentage:.1f}%)")
    
    # 2. Text length insights
    avg_spam_length = df[df['label'] == 1]['message_length'].mean()
    avg_ham_length = df[df['label'] == 0]['message_length'].mean()
    insights.append(f"Average spam message length: {avg_spam_length:.1f} characters")
    insights.append(f"Average ham message length: {avg_ham_length:.1f} characters")
    
    if avg_spam_length > avg_ham_length:
        insights.append("Spam messages tend to be longer than legitimate emails")
    else:
        insights.append("Legitimate emails tend to be longer than spam messages")
    
    # 3. Word count insights
    avg_spam_words = df[df['label'] == 1]['word_count'].mean()
    avg_ham_words = df[df['label'] == 0]['word_count'].mean()
    insights.append(f"Average spam word count: {avg_spam_words:.1f} words")
    insights.append(f"Average ham word count: {avg_ham_words:.1f} words")
    
    # 4. Model performance insights
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    insights.append(f"Best performing model: {best_model_name} (Accuracy = {best_accuracy:.3f})")
    
    if best_accuracy > 0.95:
        insights.append("Excellent model performance for spam detection")
    elif best_accuracy > 0.90:
        insights.append("Very good model performance for spam detection")
    elif best_accuracy > 0.85:
        insights.append("Good model performance for spam detection")
    else:
        insights.append("Model performance could be improved with feature engineering")
    
    # 5. Feature insights
    insights.append("TF-IDF vectorization captures word importance and frequency")
    insights.append("Bigram features help capture phrase patterns common in spam")
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Use the best performing model for production spam detection")
    print("2. Regularly retrain the model with new spam patterns")
    print("3. Consider ensemble methods for more robust detection")
    print("4. Monitor false positive rates to avoid blocking legitimate emails")
    print("5. Update the feature extraction pipeline as spam evolves")
    print("6. Implement real-time filtering based on confidence scores")

def main():
    """Main function to run the complete spam detection pipeline"""
    print("EMAIL SPAM DETECTION WITH MACHINE LEARNING")
    print("="*70)
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Clean data
        df_clean = clean_data(df)
        
        # Explore data
        explore_data(df_clean)
        
        # Prepare data for ML
        X_train, X_test, y_train, y_test, vectorizer = prepare_data(df_clean)
        
        # Train models
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        best_model_name, best_model = evaluate_models(results, y_test)
        
        # Detailed evaluation of best model
        detailed_evaluation(best_model, best_model_name, X_test, y_test, vectorizer)
        
        # Hyperparameter tuning
        tuned_model = hyperparameter_tuning(X_train, y_train)
        
        # Make predictions
        make_predictions(tuned_model, vectorizer)
        
        # Generate insights
        generate_insights(df_clean, results)
        
        print("\n" + "="*70)
        print("SPAM DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


















