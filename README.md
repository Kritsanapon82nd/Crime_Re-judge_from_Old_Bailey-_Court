# ⚖️ Historical AI Lab: Decoding the Old Bailey

An interactive Machine Learning project that reconstructs and simulates judicial decisions from the Old Bailey court (18th–19th century London), allowing users to explore how justice evolved across legal eras.

## 🧠 Project Overview

This project leverages historical crime records from the Old Bailey to build a Machine Learning system that simulates how an AI-powered “Supreme Court” might judge criminal cases under different historical legal frameworks.

Users can interact with a Streamlit web application to create **"What-If" scenarios** by modifying:

- Crime categories
- Defendant characteristics
- Contextual factors (e.g., violence, social class)

The system predicts how verdicts would differ across time periods such as:

- Early Bloody Code Era
- Late Victorian Legal System

## ⚙️ Machine Learning Approach

The project uses a **Soft Voting Ensemble Model** combining:

- Random Forest (captures non-linear relationships)
- XGBoost (high-performance gradient boosting)

### 🧩 Key Features

- End-to-end ML Pipeline (preprocessing + model)
- Label encoding for judicial outcomes
- Robust fallback system (single-model mode if ensemble unavailable)
- Historical feature engineering

## 📊 Model Performance Comparison

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| XGBoost                 | 0.667    | 0.480     | 0.470  | 0.470    |
| Random Forest           | 0.667    | 0.480     | 0.450  | 0.450    |
| Linear SVM              | 0.647    | 0.420     | 0.390  | 0.390    |
| XGBoost + Random Forest | 0.671    | 0.490     | 0.460  | 0.470    |

All models were trained and evaluated on the same dataset split to ensure a fair comparison.

The ensemble model (XGBoost + Random Forest) achieves the highest overall accuracy, suggesting that combining models improves predictive performance and stability compared to individual models.

## 🗂️ Project Structure

```text
Crime_Re-judge_From_Old_Bailey/
│
├── data/
│   ├── raw/                    # Raw dataset (Sessions Papers)
│   └── processed/              # Cleaned datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_pipeline_model_Judge.ipynb
│
├── app.py                      # Streamlit application
├── Old_bailey_xgb_pipeline.pkl # XGBoost pipeline model
├── label_encoder.pkl           # Target encoder
├── X_test.csv
├── y_test.csv
├── requirements.txt
└── README.md
```

## ⚠️ Large Model Handling (Important)

The Random Forest model file:

Old_bailey_rf_pipeline.pkl (~1.46 GB)

exceeds GitHub’s 100 MB file size limit and is therefore hosted externally.

### 📥 Full Ensemble Setup

1. Download the Random Forest model from Google Drive:
   https://drive.google.com/drive/folders/1d_Gesv6K4fmdoQLYn2D4awob4sRp6710?usp=sharing

2. Place the file in the project root directory (same level as `app.py`)

3. Run the application — the system will automatically activate Ensemble Mode

## 🔁 Fallback Mechanism

If the Random Forest model is not detected:

- The system automatically switches to XGBoost-only mode
- The application continues to function without interruption

## 💻 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Kritsanapon82nd/Crime_Re-judge_from_Old_Bailey-_Court.git
cd Crime_Re-judge_from_Old_Bailey-_Court
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

```bash
streamlit run app.py
```

## 🎮 How to Use

### 1. Fetch a Historical Case

Use the sidebar to retrieve a real Old Bailey case

### 2. Review the Base Reality

Examine:

- Crime description
- Defendant profile
- Actual historical verdict

### 3. Modify the Scenario (Time Machine)

Adjust variables such as:

- Crime severity
- Presence of violence
- Social class

### 4. Generate Verdict

Click **"Run Time Machine"** to simulate outcomes across multiple legal eras

## 🎯 Key Insights

- Legal outcomes were strongly influenced by social class and historical era
- The Bloody Code period reflects harsher sentencing patterns
- The model highlights potential bias patterns embedded in historical judicial systems
- Later legal systems show more stable and predictable decision behavior

## 🛠️ Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Streamlit

## 📚 Dataset Source

- Old Bailey Proceedings (1674–1913)
- Digitized historical court records

## 🤝 Contributing

Contributions are welcome. Feel free to fork this repository and submit a pull request.

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author
- **GitHub:** [Kritsanapon82nd](https://github.com/Kritsanapon82nd)
