# MNIST Digit Recognition Project

A comprehensive implementation of digit recognition using both Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) on the MNIST dataset. View the dataset and more info on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/code).

## Project Overview

This project implements and compares different machine learning approaches for handwritten digit recognition, achieving a best accuracy of 98.72%.

## Practical Applications

- **Postal Sorting Systems**: Automating mail sorting by reading handwritten digits
- **Banking & Check Processing**: Automating check recognition and verification
- **OCR Systems**: Converting handwritten documents into digital text
- **Assistive Technologies**: Helping visually impaired individuals read numerical information
- **Educational Tools**: Supporting handwriting recognition in educational applications

## Dataset Structure
- Training data: (42000, 785) samples
- Test data: (28000, 784) samples
- Image dimensions: 28x28 pixels (784 features when flattened)

## Data Preprocessing

### Normalization
- Images scaled by dividing by 255.0 to range [0,1]

### Format-specific preprocessing
- CNN: Data reshaped to (-1, 28, 28, 1) format
- SVM: Data flattened to 2D format

## Model Implementations

### 1. Convolutional Neural Network (CNN)
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### MNIST Digit Recognition using SVM

## Data Preprocessing
- **Training data**: (42000, 785) samples
- **Test data**: (28000, 784) samples
- **Normalization**: Images scaled by dividing by 255.0 to range [0, 1]
- **Format-specific preprocessing**: Data flattened to 2D format for SVM input

## SVM Implementation

### Configuration
```python
svm_model = SVC(
    kernel='rbf',            # Radial Basis Function kernel
    C=10.0,                 # Regularization parameter
    gamma='scale',          # Kernel coefficient
    random_state=42,
    verbose=True
)
```

### Training Details
- Training performed on subset (10,000 samples) due to computational constraints
- Hyperparameter tuning performed using GridSearchCV
- Parameters tuned:
  - C: Regularization strength
  - gamma: Kernel coefficient
  - kernel: RBF selected as optimal

## Performance Metrics
- **Validation Accuracy**: ~96% (on subset)
- Key characteristics:
  - More computationally intensive for full dataset
  - Good generalization on test set
  - Effective for digit classification task

## Limitations and Considerations
1. Computational Constraints:
   - Training on full dataset requires significant computational resources
   - Used subset of data for practical implementation

2. Memory Requirements:
   - SVM memory usage scales with dataset size
   - Requires flattened input format (784 features per image)

## Future Improvements
1. Feature extraction optimization
2. Kernel selection refinement
3. Parameter tuning for better performance
4. Implementation of multi-class strategies
5. Memory optimization techniques

This implementation demonstrates the effectiveness of SVM for digit recognition while acknowledging practical computational constraints in large-scale applications.
