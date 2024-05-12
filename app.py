from flask import Flask, render_template, request, send_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Check if a file was uploaded
    if 'dataset' not in request.files:
        return 'No file uploaded', 400

    # Get the uploaded file and target column name
    dataset_file = request.files['dataset']
    target_col = request.form['target_col'].strip()  # Strip leading and trailing whitespaces

    # Load the dataset from the uploaded file
    try:
        df = pd.read_csv(dataset_file)
    except Exception as e:
        return f'Error loading dataset: {str(e)}', 400

    # Ensure that the target column exists in the dataset
    if target_col not in df.columns:
        return f'Target column "{target_col}" not found in the dataset', 400

    # Preprocess the dataset to handle non-numeric values
    df = preprocess_dataset(df)

    # Extract features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model (example with SVM)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save the trained model
    joblib.dump(svm_model, 'trained_model.pkl')

    # Generate visualization of confusion matrix
    cm_plot = generate_confusion_matrix_plot(cm, y)

    return render_template('result.html', accuracy=accuracy, cm_plot=cm_plot)

def preprocess_dataset(df):
    # Identify columns with non-numeric values
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Convert non-numeric values to numeric using one-hot encoding
    df = pd.get_dummies(df, columns=non_numeric_cols)

    return df
# Function to generate a base64-encoded image from a Matplotlib plot
def generate_confusion_matrix_plot(cm, y):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(set(y)))
    plt.xticks(tick_marks, set(y), rotation=45)
    plt.yticks(tick_marks, set(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(set(y))):
        for j in range(len(set(y))):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Convert plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    # Encode bytes as base64
    encoded_img = base64.b64encode(img_bytes.read()).decode('utf-8')

    return encoded_img


@app.route('/download_model')
def download_model():
    return send_file('trained_model.pkl', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
