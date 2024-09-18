# Multi-label Text Classification using BERT 

## Overview

This project focuses on fine-tuning BERT (Bidirectional Encoder Representations from Transformers) for a multi-label classification task. The objective is to classify text prompts into multiple categories using a BERT-based model.

## Dependencies

Ensure the following libraries are installed. You can install them using `pip`:

```bash
pip install numpy pandas tqdm scikit-learn torch transformers wordcloud matplotlib
```

## Data

- **Training Data**: `/kaggle/input/prompts-classification/train_final_v1.csv`
- **Validation Data**: `/kaggle/input/prompts-classification/test_final_v1.csv`

## Configuration

- **Maximum Sequence Length**: 200
- **Training Batch Size**: 64
- **Validation Batch Size**: 64
- **Learning Rate**: 2e-5
- **Number of Epochs**: 100

## Workflow

### 1. Data Preparation

1. Load training and validation datasets.
2. Define target columns and initialize the tokenizer.

### 2. Custom Dataset

- Implement `Custom_Dataset` to handle tokenization and prepare data for the model.

### 3. Model Definition

- Utilize a BERT-based model with an additional linear classification layer.

### 4. Training

- Optimize using AdamW with binary cross-entropy loss.
- Save model checkpoints every 20 epochs for further analysis.

### 5. Evaluation

- Compute F1 scores (micro, macro, and weighted) on the validation set.
- Generate a classification report to summarize model performance.

### 6. Visualization

- Generate and display word clouds for each category to analyze term frequency in the dataset.

## Code Snippets

Below are key snippets of the implementation:

**Data Loading and Preparation**

```python
import pandas as pd

# Load datasets
df_train = pd.read_csv('/kaggle/input/prompts-classification/train_final_v1.csv')
df_val = pd.read_csv('/kaggle/input/prompts-classification/test_final_v1.csv')
```

**Custom Dataset Class**

```python
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    ...
```

**Model Definition**

```python
import torch.nn as nn
from transformers import AutoModel

class Custom_model(nn.Module):
    ...
```

**Training Loop**

```python
for epoch in range(EPOCHS):
    ...
```

**Evaluation and Visualization**

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word clouds
...
```

## Results

- **F1 Scores**:
  - Micro: `X%`
  - Macro: `X%`
  - Weighted: `X%`

- **Classification Report**: Comprehensive performance metrics provided in the output.

## Visualization

Word clouds for each category have been created to visualize the distribution of frequent terms in the text data.
