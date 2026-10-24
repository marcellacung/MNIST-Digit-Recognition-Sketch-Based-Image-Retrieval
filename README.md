# MNIST Digit Recognition & Sketch-Based Image Retrieval

An interactive Human-Computer Interaction (HCI) web application built with Flask. This project features two core modules: a real-time **Handwritten Digit Recognition** system comparing Convolutional Neural Networks (CNN) and Fully Connected (FC) networks, and a cross-modal **Sketch-Based Image Retrieval** system using VGG16 features.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Models & Architecture](#-models--architecture)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 📖 Overview

This project demonstrates the practical application of HCI principles, Machine Learning, and Web Development. 

1. **Task 1 & 2 (Digit Recognition)**: Users draw a digit (0-9) on an interactive canvas. The system preprocesses the drawing (centering, normalization) and predicts the digit using two models simultaneously, allowing users to compare the confidence and accuracy of a CNN vs. a standard FC network.
2. **Task 3 (Sketch Retrieval)**: Users draw a rough sketch, and the system retrieves the most visually similar real-world photos from a pre-extracted SketchyDatabase feature set using VGG16 and cosine similarity.

---

## ✨ Key Features

- ✅ **Dual-Model Comparison**: Real-time side-by-side prediction from CNN and FC models.
- ✅ **Smart Image Preprocessing**: Automatically centers (using Center of Mass) and normalizes user drawings to match the MNIST dataset.
- ✅ **Practice Mode**: Provides visual hints and tips on how to draw specific digits (especially 3, 4, and 7).
- ✅ **Sketch Retrieval**: Cross-modal retrieval (sketch to real image) using deep learning features.
- ✅ **Interactive Canvas**: Built with HTML5 Canvas and JavaScript for a smooth drawing experience.
- ✅ **Network Visualization**: Built-in diagrams explaining the CNN and FC architectures.

---

## 🗂 Project Structure

```text
MNIST-Digit-Recognition-Sketch-Based-Image-Retrieval/
│
├── app.py                        # Main Flask application
├── requirements.txt
├── demo.mp4
├── model_comparison_results.csv
│
├── data/                         # Data loaders
├── features/                     # Pre-extracted VGG features for retrieval
├── models/                       # Sketch retrieval models (ResNet/VGG)
├── models_12/                    # Trained MNIST models (CNN & FC .pkl)
├── sketchy_system/               # SketchyDatabase core logic
├── utils/                        # Helper scripts (Extractor, Visualizer)
│
├── static/                       # Web assets
│   ├── css/ & js/                # Styling and frontend logic
│   ├── hints/                    # 0-9 practice images
│   ├── network_diagrams/         # Generated architecture diagrams
│   ├── drawings/ & demo/         # User drawing outputs
│   └── sketchy_results/          # Retrieved images
│
└── templates/                    # HTML templates
    ├── index.html                # Main digit recognition UI
    ├── sketch_retrieval.html     # Sketch retrieval UI
    ├── network_diagrams.html     # Model architecture page
    └── results.html              # Model comparison page
```

---

## 🧠 Models & Architecture

### 1. Convolutional Neural Network (CNN)
- **Architecture**: Input (28x28x1) → Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → Flatten → Dense (128) → Dropout (0.5) → Output (10).
- **Total Parameters**: ~421,642
- **Accuracy**: ~95.8%
- **Advantage**: Excellent at capturing spatial hierarchies and local features in images.

### 2. Fully Connected Network (FC)
- **Architecture**: Input (784) → Dense (256) → Dense (128) → Dense (64) → Output (10).
- **Total Parameters**: ~242,762
- **Accuracy**: ~95.2%
- **Advantage**: Simpler and faster to train, but lacks spatial awareness.

### 3. Sketch Retrieval System
- Uses a pre-trained **VGG16** model to extract features from both the user's sketch and the database photos.
- Computes **Cosine Similarity** to rank and return the top-K most similar real images.

---

## 🚀 How to Run

### Prerequisites
1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure `torch`, `torchvision`, `scikit-learn`, `tensorflow`, `Pillow`, `scipy`, `numpy`, `pandas`, and `matplotlib` are installed).*

### Execution
1. **Clone the repository**
   ```bash
   git clone https://github.com/marcellacung/MNIST-Digit-Recognition-Sketch-Based-Image-Retrieval.git
   cd MNIST-Digit-Recognition-Sketch-Based-Image-Retrieval
   ```

2. **Start the Flask server**
   ```bash
   python app.py
   ```

3. **Access the Web Application**
   Open your browser and navigate to:
   - **Home (Digit Recognition)**: `http://localhost:8080`
   - **Sketch Retrieval**: `http://localhost:8080/sketch_retrieval`
   - **Network Diagrams**: `http://localhost:8080/network_diagrams`
   - **Model Comparison**: `http://localhost:8080/compare`

*(Note: Since the models and features are included in this repository, the application will load them directly without needing to re-train).*

---

## 🛠 Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras (CNN), Scikit-Learn (FC/MLP), PyTorch (VGG16 for retrieval)
- **Frontend**: HTML5 Canvas, JavaScript, CSS3
- **Image Processing**: PIL, SciPy (Gaussian Filter, Center of Mass)
- **Data Visualization**: Matplotlib

---

## 👤 Author

**MARCELLA**
- Institution: Shantou University
- Major: Data Science and Big Data Technology 
- Course: Human-Computer Interaction (HCI)
- Email: 23marcella@stu.edu.cn

---

## 📄 License

This project is created for educational purposes as part of the HCI course final project.
