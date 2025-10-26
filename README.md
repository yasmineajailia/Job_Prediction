# 🎯 Job Prediction Based on Skills

A deep learning project that predicts job roles based on skills using Bidirectional LSTM neural networks. This project leverages Natural Language Processing (NLP) to analyze skill sets and classify them into appropriate job categories.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project uses a Bidirectional LSTM model to predict suitable job roles based on a given set of skills. The model is trained on a dataset containing various job roles and their associated skills, making it useful for:
- Job recommendation systems
- Career path guidance
- Skills gap analysis
- Resume screening automation

## ✨ Features

- **Data Visualization**: 
  - Missing values heatmap
  - Job role frequency distribution
  - Skills distribution analysis
  - Word cloud visualization of skills

- **Deep Learning Model**:
  - Bidirectional LSTM architecture
  - Embedding layer for text representation
  - Dropout and SpatialDropout for regularization
  - Early stopping to prevent overfitting

- **Text Processing**:
  - Tokenization with configurable vocabulary size
  - Sequence padding for uniform input length
  - Label encoding for multi-class classification

- **Model Evaluation**:
  - Test accuracy metrics
  - Prediction examples
  - Real-time job prediction from skill descriptions

## 📊 Dataset

The project uses the [roles-based-on-skills](https://huggingface.co/datasets/fazni/roles-based-on-skills) dataset from Hugging Face, which contains:
- Job roles mapped to required skills
- Training and test splits
- Multiple job categories
- Comprehensive skill descriptions

## 🏗️ Model Architecture

The BiLSTM model consists of:

\\\
- Embedding Layer (10,000 words, 200 dimensions)
- SpatialDropout1D (0.3)
- Bidirectional LSTM (128 units, return_sequences=True)
- Bidirectional LSTM (64 units)
- Dense Layer (128 units, ReLU activation)
- Dropout (0.4)
- Output Dense Layer (num_classes, Softmax activation)
\\\

**Hyperparameters:**
- Vocabulary Size: 10,000 words
- Maximum Sequence Length: 50
- Embedding Dimension: 200
- Batch Size: 64
- Training Epochs: 15 (with early stopping)
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
\\\ash
git clone https://github.com/yasmineajailia/Job_Prediction.git
cd Job_Prediction
\\\

2. Install required packages:
\\\ash
pip install pandas numpy scikit-learn tensorflow keras wordcloud seaborn matplotlib
\\\

Or use requirements.txt (if available):
\\\ash
pip install -r requirements.txt
\\\

## 💻 Usage

### Running the Notebook

1. Open the Jupyter notebook:
\\\ash
jupyter notebook JobPrediction.ipynb
\\\

2. Run all cells sequentially to:
   - Load and explore the data
   - Visualize data distributions
   - Train the BiLSTM model
   - Evaluate model performance
   - Make predictions

### Making Predictions

You can use the \predict_job()\ function to predict job roles:

\\\python
# Example 1: Technical skills
predict_job("python machine learning tensorflow pandas numpy")
# Output: Data Scientist / ML Engineer

# Example 2: Creative skills
predict_job("Photography Graphic Design Adobe Photoshop Adobe Illustrator")
# Output: Graphic Designer / Visual Artist

# Example 3: Web development skills
predict_job("Java JavaScript Android Development PHP HTML SQL MySQL CSS")
# Output: Full Stack Developer / Web Developer
\\\

## 📈 Results

- **Test Accuracy**: ~XX% (varies based on training)
- **Number of Job Categories**: Automatically detected from dataset
- **Vocabulary Size**: 10,000 most common words
- **Training Time**: Depends on hardware (GPU recommended)

### Sample Predictions

The model successfully predicts:
- Technical roles (Data Scientist, Software Engineer, etc.)
- Creative roles (Graphic Designer, Photographer, etc.)
- Mixed skill sets (Full Stack Developer, etc.)

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Data preprocessing and evaluation
- **Matplotlib & Seaborn**: Data visualization
- **WordCloud**: Text visualization
- **Hugging Face Datasets**: Dataset source

## �� Project Structure

\\\
Job_Prediction/
│
├── JobPrediction.ipynb     # Main Jupyter notebook
├── README.md               # Project documentation
├── tokenizer.pkl          # Saved tokenizer (after training)
├── label_encoder.pkl      # Saved label encoder (after training)
├── requirements.txt       # Python dependencies (optional)
└── data/                  # Dataset directory (auto-downloaded)
    ├── train-*.parquet
    └── test-*.parquet
\\\

## 🔮 Future Improvements

- [ ] Implement attention mechanism
- [ ] Add model checkpointing
- [ ] Create Flask/FastAPI web interface
- [ ] Deploy as a REST API
- [ ] Add more visualization metrics
- [ ] Implement cross-validation
- [ ] Add support for multiple languages
- [ ] Create a skills recommendation feature

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (\git checkout -b feature/AmazingFeature\)
3. Commit your changes (\git commit -m 'Add some AmazingFeature'\)
4. Push to the branch (\git push origin feature/AmazingFeature\)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Yasmine Laajailia**

- GitHub: [@yasmineajailia](https://github.com/yasmineajailia)

## 🙏 Acknowledgments

- Dataset provided by [fazni](https://huggingface.co/fazni) on Hugging Face
- Inspired by the need for automated job-skills matching
- Built with modern NLP and deep learning techniques

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository owner.

---

⭐ If you find this project useful, please consider giving it a star!
