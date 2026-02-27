🎵 Music Emotion Recognition using CNN (PyTorch)

A deep learning project that predicts Valence and Arousal values from music audio using Convolutional Neural Networks (CNNs) trained on mel-spectrogram representations.

This project demonstrates end-to-end ML pipeline development including preprocessing, model design, training, evaluation, and deployment-ready model saving.

🚀 Key Highlights

Built a CNN regression model using PyTorch

Processed raw audio into mel-spectrogram features

Implemented label normalization for stable regression

Designed adaptive pooling architecture to reduce overfitting

Achieved strong regression performance on DEAM dataset

📊 Model Performance
Metric	Score
Test RMSE	0.2297
Test MAE	0.1949

(Values reported on normalized [-1, 1] emotion scale)

Equivalent error < 1 unit on original 1–9 DEAM scale.

🏗 Model Architecture

3 × Conv2D Blocks

Conv2D

BatchNorm

ReLU

MaxPooling

Adaptive Average Pooling

Fully Connected Regression Head

Tanh activation for bounded output

🧠 Technical Skills Demonstrated

PyTorch model building

Custom Dataset & DataLoader

Audio feature engineering (Mel Spectrograms)

Regression modeling

Model evaluation (RMSE, MAE)

Overfitting control via pooling & dropout

Modular project structure

Reproducible training pipeline

📂 Project Structure
src/            # Model, dataset, training & evaluation
notebooks/      # EDA & experimentation
models/         # Saved trained model (.pth)
results/        # Loss plots
▶️ How to Run
Install dependencies
pip install -r requirements.txt
Train model
python src/train.py
Evaluate model
python src/evaluate.py
📈 Future Enhancements

Learning rate scheduler

Early stopping

Attention-based CNN

Transformer-based audio model

Web deployment (Streamlit / Flask)

👨‍💻 Author

Harsh
B.Tech Mechanical Engineering
IIT HYD
Interested in Machine Learning & AI System 
