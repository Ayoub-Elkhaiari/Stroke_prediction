# Stroke Prediction using TensorFlow and PyTorch

## Project Overview
This project aims to predict the likelihood of a stroke using machine learning techniques. We implement and compare different approaches using TensorFlow and PyTorch, including a feedforward neural network and Google's TabNet architecture. The project showcases advanced feature engineering, data balancing techniques, and the application of state-of-the-art Deep learning models.

## Features
- Comprehensive feature engineering
- Data balancing to address class imbalance
- Implementation of a feedforward neural network using TensorFlow
- Implementation of TabNet using PyTorch


## Dataset
the famous stroke kaggle dataset : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Requirements
- Python 3.7+
- TensorFlow 2.x
- PyTorch 1.x
- pandas
- numpy
- scikit-learn




## Data Preprocessing and Feature Engineering
- Handling missing values
- Encoding categorical variables
- Feature scaling and normalization
- Implementing data balancing techniques (e.g., SMOTE, `class weighting`)

## Models

### TensorFlow Feedforward Neural Network
- Architecture: [Describe the layers and neurons]
- Activation functions: [e.g., ReLU, Sigmoid]
- Optimization: [e.g., Adam optimizer]
- Loss function: [e.g., Binary cross-entropy]

### PyTorch TabNet
- Utilizes the encoder part in attentive transformer
- Implements feature transformer for enhanced prediction
- Tabnet research paper link: https://arxiv.org/pdf/1908.07442
- i used just the encoder part to make predections :
  
  ![Screenshot 2024-09-18 205329](https://github.com/user-attachments/assets/7ca654e6-9b2b-4520-b9a0-3dbe8a3cb856)

- and if your goal maybe to handle the missing values you can use the whole enoder-decoder part:
  
  ![Screenshot 2024-09-18 205346](https://github.com/user-attachments/assets/16c4130d-7365-4669-a09c-c3c1196f87d5)

- and this is how it worked behind the scene:
  
  ![Screenshot 2024-09-18 205409](https://github.com/user-attachments/assets/d63df7e3-6261-4ff7-83f9-e4ebe7590312)

  

## Results
- For Tensorflow

  ![Screenshot 2024-09-18 205641](https://github.com/user-attachments/assets/7686dabf-7c18-42e1-a478-0d4948cd45ae)
  
  ![Screenshot 2024-09-18 205650](https://github.com/user-attachments/assets/74c031d8-01c3-4181-af45-4267ae9c8b9f)
  
  ![Screenshot 2024-09-18 205658](https://github.com/user-attachments/assets/95d682df-b257-4322-aa9a-a3603c11972a)

- For TabNet with Pytorch
  
  ![Screenshot 2024-09-18 205509](https://github.com/user-attachments/assets/d00dc7ca-0ac3-4437-89b3-44ccf49ed945)
  
  ![Screenshot 2024-09-18 205520](https://github.com/user-attachments/assets/05585c1a-cec0-4661-9290-c87e48bee70c)
  
  ![Screenshot 2024-09-18 205532](https://github.com/user-attachments/assets/6f0a89d0-7ce2-48d6-ac3e-2e3b01c64b7f)

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/Ayoub-Elkhaiari/Stroke_prediction.git
   cd Stroke_prediction
   ```

2. Install dependencies:
  - tensorflow
  - pytorch
  - tabnet classifier
  - sklearn
  - numpy
  - pandas



## Key Findings
- Comparison of model performances (TabNet vs Feedforward NN)
- Impact of feature engineering on prediction accuracy
- Effectiveness of data balancing techniques

## Future Work
- Ensemble methods combining different models
- Exploration of other advanced architectures
- Deployment of the best-performing model as a web service


## Acknowledgements
- Google Cloud AI team for the TabNet architecture


