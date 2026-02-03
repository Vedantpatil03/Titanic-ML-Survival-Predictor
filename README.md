## Titanic Survival Prediction using Machine Learning

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques. The dataset is preprocessed, analyzed, and a classification model is trained to make survival predictions.

📌 Project Overview

The sinking of the Titanic is one of the most famous shipwrecks in history. In this project, we build a machine learning model that learns from passenger data such as age, gender, class, and fare to predict survival.

🧠 Machine Learning Workflow

Load the Titanic dataset

Perform Exploratory Data Analysis (EDA)

Handle missing values

Encode categorical features

Split data into training and testing sets

Train a Random Forest Classifier

Evaluate model performance

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook / Google Colab

📂 Dataset

The dataset used is Titanic-Dataset.csv, which contains the following features:

PassengerId

Survived

Pclass

Name

Sex

Age

SibSp

Parch

Ticket

Fare

Cabin

Embarked

⚙️ Model Used

Random Forest Classifier

The model is trained using:

train_test_split for data splitting

RandomForestClassifier for prediction

Accuracy score for evaluation

📊 Data Preprocessing

Missing values handled using mean/mode

Categorical variables converted to numerical form

Unnecessary columns removed

Data cleaned for model training

📈 Model Evaluation

The model performance is evaluated using:

Accuracy Score

Classification metrics

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/titanic-survival-prediction.git


Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Open the notebook:

jupyter notebook


Run all cells in:

titanic_sets.ipynb

📌 Results

The trained machine learning model predicts passenger survival based on input features and achieves good accuracy on test data.
