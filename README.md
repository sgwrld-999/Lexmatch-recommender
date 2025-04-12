# CLATConnect

A simple, human-friendly AI/ML system that recommends the *right mentors* (CLAT toppers) to *law aspirants*, based on their preparation level, learning preferences, and interaction history. Built as part of my NLTI Internship Assignment.

---

## ✅ Objective

> Design a hybrid AI/ML model that matches aspirants with suitable mentors to boost exam prep, using both content and collaborative filtering.

---

## 📦 Dataset

Generated synthetic data for:
- `aspirants.csv` – 100 aspirants (with preferences, styles, test scores, etc.)
- `mentors.csv` – 50 mentors (experience, teaching style, mentorship history)
- `interactions.csv` – Sparse logs of aspirant-mentor engagements

Processed versions:  
- `aspirants_processed.csv`  
- `mentors_processed.csv`

---

## 🧠 What’s Inside?

### 1. **Feature Engineering**
- One-hot encoded `Preferred Subjects`, `Target Colleges`, `Learning Styles`
- Ordinal mapping of `Preparation Level`
- Numeric mapping of `Subject Strengths & Weaknesses`
- Mentor features encoded similarly

### 2. **Hybrid Recommendation System**
- **Content-Based Filtering**  
  - Computes similarity between aspirant and mentor profiles (based on subjects, learning-teaching styles)
- **Collaborative Filtering**  
  - Shallow neural model using TensorFlow/Keras  
  - Learns from past interaction ratings (if any)
- **Fusion Layer**  
  - Combines both to predict mentor-aspirant fit score

### 3. **Overfitting Mitigation**
- `Dropout`, `L2 Regularisation`, `EarlyStopping`  
- Ratings normalised during training

### 4. **Explainability (Bonus)**
- SHAP analysis shows why a mentor was recommended (factors like profile match, interaction weight)

---

## 🔧 How To Run

```bash
pip install pandas numpy tensorflow scikit-learn shap matplotlib
