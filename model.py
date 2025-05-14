import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from bs4 import BeautifulSoup
import urllib.parse
import joblib

# Load the dataset
df = pd.read_csv('phishing_email.csv')

# Assuming the dataset has 'text' and 'label' columns (phishing=1, legitimate=0)
# If column names are different, adjust accordingly

# Feature Extraction Functions
def extract_features(email_text):
    features = {}
    
    # 1. Lexical Features
    features['length'] = len(email_text)
    features['num_words'] = len(email_text.split())
    features['num_chars'] = len(email_text.replace(" ", ""))
    features['avg_word_length'] = features['num_chars'] / max(1, features['num_words'])
    features['num_capitals'] = sum(1 for c in email_text if c.isupper())
    features['capitals_ratio'] = features['num_capitals'] / max(1, features['num_chars'])
    
    # 2. HTML Features
    try:
        soup = BeautifulSoup(email_text, 'html.parser')
        features['has_html'] = 1 if len(soup.find_all()) > 0 else 0
        features['num_links'] = len(soup.find_all('a'))
        features['num_images'] = len(soup.find_all('img'))
    except:
        features['has_html'] = 0
        features['num_links'] = 0
        features['num_images'] = 0
    
    # 3. URL Features
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
    features['num_urls'] = len(urls)
    features['has_url'] = 1 if features['num_urls'] > 0 else 0
    
    url_features = []
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        url_features.append(len(parsed.path))
        url_features.append(len(parsed.query))
        url_features.append(1 if '@' in parsed.netloc else 0)  # Suspicious email in domain
        url_features.append(parsed.netloc.count('.'))
    
    features['avg_url_path_length'] = np.mean(url_features[::3]) if url_features else 0
    features['avg_url_query_length'] = np.mean(url_features[1::3]) if url_features else 0
    features['url_has_at'] = max(url_features[2::3]) if url_features else 0
    features['avg_url_dots'] = np.mean(url_features[3::3]) if url_features else 0
    
    return features

print("Feature Extraction...")
# Extract features for all emails
feature_list = []
for text in df['text_combined']:
    feature_list.append(extract_features(text))

features_df = pd.DataFrame(feature_list)

# Add TF-IDF features for text content
tfidf_content = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = tfidf_content.fit_transform(df['text_combined'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

# Combine all features
final_features = pd.concat([features_df, tfidf_df], axis=1)

# Prepare data for modeling
X = final_features
y = df['label']  # Assuming 1 for phishing, 0 for legitimate

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Save model and Vectorisers
joblib.dump(model, 'phishingModel.pkl')
joblib.dump(tfidf_content, 'tfidf_content.pkl')
