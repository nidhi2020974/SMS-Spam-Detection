# 📧 Spam vs Ham Message Detector

A web app built with **Streamlit** and **Python** to detect whether a message is **spam** or **ham (normal)** using a machine learning model.

---

## **Project Overview**

This project uses the **Naive Bayes classifier** with **TF-IDF vectorization** to classify SMS messages as spam or ham. It handles imbalanced data by upsampling the minority class (spam) to improve prediction accuracy.

Users can interactively type messages in a **Streamlit web interface** and instantly see the prediction.

---


## **Features**

- Real-time spam detection
- Simple and interactive web interface
- Handles imbalanced datasets
- Uses a classic ML algorithm (Naive Bayes) with TF-IDF features

---

## **Dataset**

The dataset is from the **Kaggle Spam SMS collection**:

- `spam.csv`
- Columns:
  - `v1` → Label (`ham` or `spam`)
  - `v2` → Message text
- Preprocessing includes selecting relevant columns and encoding labels (`ham = 0`, `spam = 1`)

---

