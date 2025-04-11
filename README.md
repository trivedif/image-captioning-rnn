# Image Caption Generation with RNNs (LSTM & GRU)

This project explores generating descriptive image captions using deep learning. It integrates pre-trained CNNs for image feature extraction with RNN-based sequence generators (LSTM and GRU) to predict natural language captions. The project uses the Flickr8k dataset and evaluates outputs using BLEU scores and semantic comparison with LLM-generated captions.

---

## Overview

- **Goal**: Automatically generate accurate captions for images using CNN-RNN pipelines.
- **Architecture**: Use InceptionV3 for visual features + LSTM (with attention) and GRU for sequence decoding.
- **Evaluation**: Compare generated captions to ground-truth using BLEU scores and LLM-based semantic similarity.
- **Tools**: PyTorch, TensorFlow/Keras, Pinecone, OpenAI API, NLTK, Matplotlib

---

## Project Highlights

- Trained and compared two captioning models:
  - **LSTM with Additive Attention and Skip Connections**
  - **GRU with Feature Fusion**
- Used **InceptionV3** for feature extraction
- Achieved **BLEU-4 score of 0.52** with LSTM
- GRU improved **inference speed by ~20%**
- Evaluated caption quality with both **BLEU scores** and **semantic similarity** using an OpenAI LLM
- Handled token padding, word embeddings, masking, and fusion for efficient training

---

## Dataset: Flickr8k

- 8,000 real-world images
- 40,000 human-annotated captions (5 per image)
- Subjects: animals, people, outdoor activities
- [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## Architecture Details

###  LSTM Model
- Input: image embeddings + token sequences
- Components: Embedding → LSTM → Additive Attention → Fusion → Dense
- Total params: 5.15M

###  GRU Model
- Input: Reduced image vectors + embedded sequences
- Components: Dense → GRU → Concatenation → Dense
- Faster than LSTM with comparable caption quality

---

## Evaluation Summary

| Metric | LSTM | GRU |
|--------|------|-----|
| BLEU-1 | 0.63 | 0.49 |
| BLEU-2 | 0.55 | 0.38 |
| BLEU-3 | 0.49 | 0.31 |
| BLEU-4 | 0.47 | 0.23 |
| Inference Speed | Baseline | +20% faster |
| Semantic Match (LLM) | High | Moderate |

---

## Project Structure

| File | Description |
|------|-------------|
| `eda_flickr8k.ipynb` | Caption length, token distribution, bigram frequency |
| `feature_visualization.ipynb` | RGB distribution & heatmaps from Flickr8k images |
| `captioning_lstm.ipynb` | LSTM model with attention, BLEU + semantic evaluation |
| `captioning_gru.ipynb` | GRU model pipeline with evaluation |
| `CaptionGeneration.pdf` | Slide deck summarizing architecture, results, and insights |

---

## How to Run

> Compatible with Jupyter, Colab, or local environments using Python 3.9+

1. Start with `eda_flickr8k.ipynb` and `feature_visualization.ipynb` to understand the data
2. Train and evaluate models via:
   - `captioning_lstm.ipynb`
   - `captioning_gru.ipynb`
3. BLEU and semantic comparison steps are included in the notebooks

Dependencies: PyTorch, TensorFlow/Keras, NLTK, Matplotlib, optionally Pinecone or OpenAI API for embeddings.

---

## Presentation Slides

For an illustrated summary of the project, including diagrams, sample captions, BLEU/LLM results, and future work:

 [CaptionGeneration.pdf](./CaptionGeneration.pdf)

---

## Authors

**Foram Trivedi**, Shashwat Shahi, Aditya Ranjan Singh  
Deep Learning · Image Captioning · BLEU · CLIP · PyTorch  
[GitHub Profile – Foram](https://github.com/trivedif)

---

## Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [OpenAI](https://openai.com/)
- [BLEU Score – NLTK](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
- [InceptionV3 – Keras Docs](https://keras.io/api/applications/inceptionv3/)
