from flask import Flask, render_template, send_file
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import io
import base64



app = Flask(__name__)

# Load the Iris dataset
dataset = load_iris()
X, y = dataset.data, dataset.target

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
    svm_model = GridSearchCV(SVC(), param_grid, cv=3)
    svm_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    cm = confusion_matrix(y_test, y_pred)

    # Save the trained model
    joblib.dump(svm_model, 'trained_model.pkl')

    # Generate visualization of confusion matrix
    cm_plot = generate_confusion_matrix_plot(cm)

    return render_template('result.html', accuracy=accuracy, cm_plot=cm_plot)

# Function to generate a base64-encoded image from a Matplotlib plot
def generate_confusion_matrix_plot(cm):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(dataset.target_names))
    plt.xticks(tick_marks, dataset.target_names, rotation=45)
    plt.yticks(tick_marks, dataset.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(dataset.target_names)):
        for j in range(len(dataset.target_names)):
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
