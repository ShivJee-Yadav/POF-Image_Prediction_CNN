# 🧠 FiberFreqNet — Image‑Driven Frequency Estimation  

<div align="center">
**Leveraging Convolutional Neural Networks for robust image‑based Probability of Failure prediction and analysis**

</div>

## 📖 Overview

This repository presents a deep learning project focused on estimating **vibration frequency (Hz)** directly from grayscale fiber images using a Convolutional Neural Network (CNN). The project automates the process of mapping raw image data to a continuous frequency target, enabling accurate prediction across a wide range (1 Hz – 22 kHz).

It includes the CNN model architecture, scripts for training and testing, data preprocessing utilities, and pre-trained model weights. The outputs cover training logs, saved models, and evaluation results to assess the model’s performance.

---

## ✨ Features

- **Image-based Frequency Regression**: Trains a CNN to predict vibration frequency values from fiber images.
- **Modified ResNet-18 Architecture**: Adapts the first convolution layer for grayscale input and replaces the final layer with a regression head.
- **Data Loading & Preprocessing**: Loads sweep and discrete datasets from CSV labels, applies resizing and normalization for model training.
- **Frequency-Dependent Oversampling**: Balances sweep (1–1000 Hz, approximate labels) and discrete (1–22 kHz, accurate labels) data for stable learning.
- **Model Training**: Provides scripts for training with early stopping, Adam optimizer, and MSE loss.
- **Performance Evaluation**: Logs training loss per epoch and saves the best-performing model weights (`frequency_regression_cnn.pth`).
- **Pre-trained Models**: Includes `.pth` weights for immediate inference and testing without re-training.

---

## 📂 Repository Structure

- `train.py` – Training script with oversampling, early stopping, and model saving.
- `test.py` – Inference script to predict frequency from new images.
- `CNN/all_labels.csv` – Dataset labels (sweep + discrete).
- `Final_Sweep_Reading/` – Image dataset directory.
- `frequency_regression_cnn.pth` – Pre-trained model weights.

---

## 🛠️ Tech Stack

**Core Machine Learning:**
-   ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
-   ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
-   ![Torchvision](https://img.shields.io/badge/Torchvision-FF8C00?style=for-the-badge&logo=pytorch&logoColor=white) (Inferred)

**Data Handling & Analysis:**
-   ![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) (Inferred for CSV handling)
-   ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) (Inferred for numerical operations)
-   ![Matplotlib](https://img.shields.io/badge/Matplotlib-366576?style=for-the-badge&logo=matplotlib&logoColor=white) (Inferred for `plotname.pdf`)
-   ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) (Potentially for metrics or utilities)

## 🚀 Quick Start

### Prerequisites
Before you begin, ensure you have the following installed:
-   **Python 3.x**
-   **pip** (Python package installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ShivJee-Yadav/POF-Image_Prediction_CNN.git
    cd POF-Image_Prediction_CNN
    ```

2.  **Install dependencies**
    It is highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the necessary Python packages
    # (A requirements.txt file is not provided, so install manually)
    pip install torch torchvision pandas numpy matplotlib scikit-learn
    # You might need a specific CUDA-enabled PyTorch version if using a GPU.
    # Refer to https://pytorch.org/get-started/locally/ for details.
    ```

### Dataset Setup
The project expects image data and labels. Based on the directory structure:
-   Image data should be organized within the `CNN/`, `NOrmal/`, and `SpinVIew/` directories.
-   Labels for the images are expected to be in `all_labels.csv`. Ensure this CSV file correctly maps to your image filenames.

*(Without specific data loading scripts, the exact expected structure for images within `CNN/`, `NOrmal/`, `SpinVIew/` is inferred. Typically, these would contain subdirectories for different classes or be structured in a way that allows a custom dataset loader to process them.)*

### Usage

#### 1. Training the Model
To train the CNN model, you would typically run a Python script.
*(A dedicated training script like `train.py` is not explicitly listed in the top-level files but would be expected within `CNN/` or `advanced_image_analysis/`.)*

```bash

# Example: If a 'train.py' script exists in the root or a subdirectory
python your_training_script.py --data_path ./ --labels_file all_labels.csv --epochs 50 --batch_size 32
```
This script would save the trained model weights (e.g., `frequency_regression_cnn.pth`) and potentially generate performance metrics in `POF_full_metrics.csv` and plots in `analysis_results/` or `plotname.pdf`.

#### 2. Performing Inference (Prediction)
You can use the provided pre-trained models (`frequency_regression_cnn.pth` or `frequency_regression_cnn_log.pth`) to make predictions on new images.
*(An inference script like `predict.py` would be needed.)*

```bash

# Example: If a 'predict.py' script exists
python your_prediction_script.py --model_path frequency_regression_cnn.pth --image_folder ./new_images --output_file predictions.csv
```

#### 3. Analyzing Results
The project provides `POF_full_metrics.csv` for quantitative analysis and `plotname.pdf` for visual insights. The `analysis_results/` directory is likely where other generated plots or processed data are stored.

## 📁 Project Structure

```
POF-Image_Prediction_CNN/
├── .gitignore               # Specifies intentionally untracked files to ignore
├── CNN/                     # Likely contains CNN model definitions, training scripts, or a subset of image data
├── LICENSE                  # MIT License file
├── NOrmal/                  # Potentially a directory for "Normal" class images or data
├── POF_full_metrics.csv     # Comprehensive CSV file containing full model performance metrics
├── README.md                # Project README file (this file)
├── SpinVIew/                # Could be related to a specific type of image view or data subset
├── Summary.txt              # A text file summarizing project details, model, or results
├── advanced_image_analysis/ # Scripts or notebooks for deeper image analysis or feature engineering
├── all_labels.csv           # CSV file containing all labels corresponding to the image dataset
├── analysis_results/        # Directory to store generated analysis outputs (plots, metrics, processed data)
├── frequency_regression_cnn.pth      # Pre-trained PyTorch model weights for frequency regression
├── frequency_regression_cnn_log.pth  # Another set of pre-trained PyTorch model weights (possibly with log transformation)
└── plotname.pdf             # A generated PDF plot, likely visualizing model performance or data characteristics
```

## ⚙️ Configuration

This project's configuration primarily relies on command-line arguments passed to the Python scripts (e.g., `--epochs`, `--batch_size`, `--data_path`). There are no external configuration files (like `.env` or `config.yaml`) explicitly detected in the top-level directory.

## 🤝 Contributing

We welcome contributions to this project! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

### Development Setup for Contributors
1.  Fork the repository.
2.  Clone your forked repository.
3.  Create a virtual environment and install dependencies as described in the [Installation](#installation) section.
4.  Implement your changes.
5.  Ensure any new code adheres to best practices and is well-documented.
6.  Submit a pull request with a clear description of your changes.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   **PyTorch**: For providing a powerful and flexible deep learning framework.
-   **NumPy & Pandas**: Essential libraries for numerical operations and data manipulation.
-   **Matplotlib**: For robust data visualization capabilities.
-   [ShivJee-Yadav](https://github.com/ShivJee-Yadav): The original author and maintainer of this project.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/ShivJee-Yadav/POF-Image_Prediction_CNN/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful for your image prediction tasks!**

Made with ❤️ by [ShivJee-Yadav](https://github.com/ShivJee-Yadav)

</div>

