# TRAVEL-INSURANCE-PREDICTION
🧠 Overview

This project predicts whether a customer will purchase travel insurance using machine learning classification techniques. The model helps insurance companies identify potential customers likely to buy travel insurance, improving targeted marketing and business insights.

📂 Project Structure travel_insurance_project/ │ ├── data_utils.py # Loads or generates dataset ├── preprocessing.py # Data cleaning & feature pipeline ├── models.py # Model training, comparison & evaluation ├── main.py # End-to-end script ├── notebooks/ │ └── Travel_Insurance_Prediction_Complete.ipynb ├── outputs/ # Model metrics & predictions ├── sample_inputs.csv # Example data for predictions ├── requirements.txt # Required libraries └── README.md # Project documentation

⚙️ Tools & Libraries Used

Languages: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

Tools: Jupyter Notebook, VS Code, Git

🔍 Methodology

Data Collection & Loading

Reads dataset (TravelInsurancePrediction.csv)

If unavailable, generates a synthetic dataset for demonstration

Data Preprocessing

Handles missing/outlier values

Encodes categorical variables

Scales numerical features

Exploratory Data Analysis (EDA)

Visualizes relationships (Age, Income, Travel History, etc.)

Correlation heatmaps & distributions

Model Building

Trained & compared three models:

Logistic Regression

Random Forest

Gradient Boosting

Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Confusion Matrix for best model

Predictions & Output

Generates sample_predictions.csv

Saves trained models (.joblib) and metrics (.json)

📊 Results & Insights

Best Model: Gradient Boosting Classifier

Accuracy: ~87% (varies slightly depending on dataset)

Key Influencing Factors:

Age

Annual Income

Frequent Flyer Status

Travelled Abroad History

Customers who travel frequently or have higher income are more likely to buy travel insurance.

🚀 How to Run

Clone the repository
git clone https://github.com//TRAVEL-INSURANCE-PREDICTION
.git cd travel-insurance-prediction

Install dependencies
pip install -r requirements.txt

Run the main pipeline
python main.py

To use your own dataset: Place your CSV file named TravelInsurancePrediction.csv in the project root.

🧩 Sample Predictions

The script saves 10 random predictions with probabilities to:

outputs/sample_predictions.csv

🏆 Future Improvements

Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

Implement SHAP or LIME for model explainability

Handle class imbalance with SMOTE

Deploy with Streamlit or Flask for live prediction

👨‍💻 Author

PRAHADHEESH.S || prahasenthuran@gmail.com

💡 Data Science & Analytics Enthusiast | Machine Learning | Python Developer
