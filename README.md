# Sentiment Analysis for Social Media Data


This project focuses on building a sentiment analysis model that can classify social media posts into positive, negative, or neutral sentiment categories. The model is trained on a labeled dataset of social media posts, such as tweets, and evaluated for its performance.

## Dataset

The dataset used for this project is a collection of tweets from various social media platforms. It contains a set of text data along with corresponding sentiment labels (0 for negative, 1 for positive). The dataset provides a representative sample of real-life social media sentiments.

## Project Structure

The project is organized as follows:

- `data`: This folder contains the dataset file (`tweets_dataset.csv`) used for training and evaluation.

- `preprocessing`: This part of the script performs data preprocessing steps, including removing URLs, punctuation, and converting text to lowercase. It prepares the dataset for further processing.

- `model`: This part of the script builds the sentiment analysis model using a deep learning approach. It defines the architecture of the model, including embedding layers, LSTM layers, and a dense layer for sentiment classification.

- `train`: This part of the script trains the sentiment analysis model on the preprocessed dataset. It splits the data into training and testing sets, performs feature engineering using TF-IDF vectorization, and trains the model using the training data.

- `evaluate`: This part of the script evaluates the performance of the trained model on the testing data. It calculates metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness in sentiment classification.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- tensorflow

## Usage

1. Clone the repository
2. Install requirements
3. Provide data file
4. Run app.py


