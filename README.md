# Brain Tumor Detection with DenseNet-121

🧠 **Brain Tumor Classification System**

This Gradio application uses a DenseNet-121 deep learning model to detect and classify brain tumors from MRI scan images. The model can identify three different types of brain conditions with high accuracy.

## 🎯 Model Performance

This DenseNet-121 model achieves **94% overall accuracy** on the test dataset:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Brain Glioma | 100% | 93% | 97% | 198 |
| Brain Meningioma | 88% | 94% | 91% | 193 |
| Brain Tumor | 95% | 95% | 95% | 214 |
| **Overall Accuracy** | | | **94%** | **605** |

## 🔬 Classification Categories

The model can detect and classify three types of brain conditions:

1. **Brain Glioma** - A type of tumor that occurs in the brain and spinal cord
2. **Brain Meningioma** - A tumor that arises from the meninges (membranes surrounding the brain)
3. **Brain Tumor** - General brain tumor classification

## 🚀 How to Use

1. **Upload an MRI Image**: Click on the upload area or drag and drop your brain MRI scan
2. **Get Prediction**: The model will automatically analyze the image and provide:
   - Predicted class with confidence score
   - Probability distribution across all three categories
3. **View Results**: Results are displayed instantly with clear visualization

## 🏗️ Model Architecture

- **Base Model**: DenseNet-121
- **Framework**:  PyTorch
- **Input Size**: 512x512 pixels (RGB)
- **Classes**: 3 (Brain Glioma, Brain Meningioma, Brain Tumor)

## ⚠️ Important Disclaimers

- **This tool is for educational and research purposes only**
- **Not intended for clinical diagnosis or medical decision-making**
- **Always consult qualified medical professionals for proper diagnosis**
- **Results should be interpreted by trained radiologists or medical experts**

## 🛠️ Technical Details

```python
# Model specifications
- Architecture: DenseNet-121
- Input Shape: (512, 512, 3)
- Output Classes: 3
- Loss Function: Categorical Crossentropy
```

## 📊 Performance Metrics

- **Macro Average**: 94% across all metrics
- **Weighted Average**: 94% across all metrics
- **Best Performing Class**: Brain Glioma (97% F1-score)
- **Most Balanced Performance**: Brain Tumor (95% across all metrics)

## 🔄 Model Updates

The model is continuously being improved. Current version shows excellent performance with:
- High precision across all classes (88-100%)
- Strong recall performance (93-95%)
- Balanced F1-scores (91-97%)

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting issues or bugs
- Suggesting improvements
- Providing feedback on model performance
- Contributing additional training data (with proper permissions)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.