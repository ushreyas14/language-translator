# Language Translator

A language translation project that translates Akkadian text to English using a fine-tuned T5 transformer model.

## About

This project implements a neural machine translation system for translating ancient Akkadian language to modern English. The model is built using the T5 (Text-To-Text Transfer Transformer) architecture and trained on a dataset of Akkadian-English text pairs.

## Features

- **Akkadian to English Translation**: Translates ancient Akkadian text to modern English
- **T5 Model**: Uses state-of-the-art transformer architecture
- **Jupyter Notebook**: Interactive notebook for training and inference
- **Evaluation Metrics**: Includes BLEU score evaluation for translation quality

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Datasets
- Evaluate
- Sacrebleu
- Pandas
- NumPy

## Usage

1. Install the required dependencies:
```bash
pip install torch transformers datasets evaluate sacrebleu pandas numpy
```

2. Open the Jupyter notebook:
```bash
jupyter notebook deep-past-challenge-2.ipynb
```

3. Follow the notebook cells to train the model or perform translations

## Model

The project uses the T5 model for sequence-to-sequence translation, fine-tuned specifically for Akkadian-to-English translation tasks.

## License

This project is open source and available for educational and research purposes.
