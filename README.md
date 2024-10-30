# English-French-Translation-Model
This project implements a translation model that converts English sentences to French, leveraging transformer-based language representations for feature extraction. The model is built using TensorFlow and Hugging Face's transformer library. It employs a pre-trained BERT model for feature extraction and a custom-trained model for translation. 

#Project Overview
Translation between languages requires understanding both sentence structure and contextual meaning, so this project leverages transfer learning with BERT's language modeling capabilities. Using BERT as a feature extractor enables the model to generate high-quality sentence embeddings that capture linguistic nuances. These features are fed into a translation model, trained specifically for English-to-French translation.

#Dataset
The dataset used for this project contains English and French sentence pairs. Each English sentence is preprocessed and passed through the transformer model to extract features. These features serve as the basis for the translation model's learning process.

#Model Architecture
Feature Extraction with BERT:

Pre-trained Model: The project uses the "bert-base-uncased" model from Hugging Face as a feature extractor. BERT’s attention mechanism captures the context in each sentence, making it ideal for extracting nuanced features.
Mean Pooling of Last Hidden State: To get a fixed-size vector for each sentence, mean pooling is applied over the last hidden states of BERT's output.
Tokenizer: Each sentence is tokenized and padded to ensure uniform length, allowing batch processing and efficient model inference.
Translation Model:

Encoder-Decoder Architecture: The custom translation model is an encoder-decoder setup that takes the extracted BERT features as input.
Encoder: The BERT features are further processed to understand sentence structure.
Decoder: The decoder learns to map the encoded English representation to French, producing accurate translations.
Training: The model is trained using sparse categorical cross-entropy, which is effective for sequence-to-sequence models, and teacher forcing is applied to stabilize training.
Installation & Requirements
Install necessary libraries:

bash
Copy code
pip install tensorflow transformers pandas numpy
Code Walkthrough
1. Data Preprocessing & Tokenization
The dataset is loaded, and the English sentences are tokenized using the BERT tokenizer. Each sentence is padded to a uniform length to facilitate batch processing.

2. Feature Extraction
The English sentences are passed through the pre-trained BERT model, and the mean of the last hidden states is taken to form a fixed-size vector representing each sentence.

3. Model Training
The translation model is trained using the extracted features and the French target sentences. An early stopping callback and checkpointing are used to prevent overfitting and save the best model weights.
