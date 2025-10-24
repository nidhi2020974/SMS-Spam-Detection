import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

# ---------- Load Dataset ----------
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only relevant columns
df = df[['v1','v2']]
df.columns = ['label','text']

# Encode labels: ham=0, spam=1
df['label_num'] = df['label'].map({'ham':0,'spam':1})

# ---------- Handle Class Imbalance ----------
df_ham = df[df.label_num==0]
df_spam = df[df.label_num==1]

# Upsample spam messages to balance dataset
df_spam_upsampled = resample(df_spam,
                             replace=True,
                             n_samples=len(df_ham),
                             random_state=42)

# Combine balanced data
df_balanced = pd.concat([df_ham, df_spam_upsampled])

# ---------- Train Naive Bayes Model ----------
X_train, X_test, y_train, y_test = train_test_split(df_balanced['text'],
                                                    df_balanced['label_num'],
                                                    test_size=0.2,
                                                    random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------- Streamlit UI ----------
st.title("📧 Spam vs Ham Message Detector")
st.markdown("Type any message below and check if it is **Spam** or **Ham**.")

# Input box for user message
user_input = st.text_area("Type your message here:")

# Check button
if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please type a message!")
    else:
        # Convert message to TF-IDF features
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        
        # Display result
        result = "Spam 🚫" if prediction==1 else "Ham ✅"
        st.success(f"Prediction: {result}")
