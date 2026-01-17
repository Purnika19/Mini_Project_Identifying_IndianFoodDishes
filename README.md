**🍽️ Indian Food Dishes Identification using Deep Learning**

This project focuses on identifying Indian food dishes from images using deep learning. Multiple models were trained and compared to understand how different CNN architectures perform on the same dataset and to automatically select the best-performing model.

**🧠 Models Trained**

1. Custom Convolutional Neural Network (CNN)

2. MobileNetV2 (Transfer Learning)

3. ResNet50 (Transfer Learning)

4. VGG16 (Transfer Learning)

The best model is selected automatically based on validation accuracy.

**📊 Dataset**

Indian Food Image Dataset: 21 food categories: Images organized class-wise : Train–validation split (80% / 20%) with stratification


**⚙️ Tech Stack**

Python , TensorFlow / Keras, NumPy, Pandas , Matplotlib, Seaborn , Google Colab

**🔄 Project Workflow**

A. Load and preprocess image data

B. Create train and validation sets

C. Train 4 different deep learning models

4. Compare training & validation accuracy

5. Plot accuracy, loss curves, and confusion matrices

6. Automatically select the best model

7. Predict food dish from uploaded images

8. Save the trained best model

**🔍 Evaluation**

1. Accuracy and loss curves for each model

2. Confusion matrix visualization

3.Best model chosen based on validation accuracy

**🎯 Purpose**

This project was built as a learning-oriented mini project to gain hands-on experience with:

I. Image classification

II. Transfer learning

III. Model comparison

IV. Practical deep learning workflows

**🚀 Future Improvements**

Fine-tuning pretrained models

A. Data augmentation

B. Deployment using Streamlit or Flask

C. Real-time prediction support
