# Anime Generator

This is a project that aims to generate realistic and coherent anime images and videos from a given story. The project uses a conditional generative adversarial network (cGAN) to produce anime images or videos from a story, and an encoder-decoder model to generate captions or dialogues for the output. The project also uses various online resources and libraries, such as Anime News Network, MyAnimeList, Anime-Planet, Kaggle, TensorFlow, and more.

## Requirements

To run this project, you will need the following:

- Python 3.8 or higher
- TensorFlow 2.6 or higher
- NumPy 1.19
- Matplotlib 3.4
- Requests 2.26
- Zipfile 3.0
- Re 2.2
- NLTK 3.6
- PIL 8.3

## Data

The data for this project consists of scripts, subtitles, and images from various anime and cartoon sources, such as web scraping the scripts, subtitles, and images from online platforms, or using existing datasets that contain anime-related content. Some possible sources are Anime News Network, MyAnimeList, Anime-Planet, and Kaggle. The data is stored in the data directory, and each file is named after the anime title.

## Code

The code for this project is written in Python and consists of the following files:

- anime_generator.py: This is the main file that contains the code for importing the libraries, defining the constants, building the models, defining the loss functions, defining the optimizers, defining the checkpoints, defining the metrics, defining the functions for data collection, data preprocessing, model training, and model evaluation.
- anime_utils.py: This is a utility file that contains some helper functions for text and image processing, such as tokenization, lemmatization, vocabulary building, encoding, decoding, resizing, cropping, augmenting, and aligning.
- anime_test.py: This is a test file that contains some code for testing the functionality and performance of the models, such as generating some samples, plotting some results, and calculating some scores.

## Usage

To run this project, you will need to follow these steps:

- Install the required libraries using pip or conda.
- Run the anime_generator.py file to collect and preprocess the data, and train and evaluate the models.
- Run the anime_test.py file to test the models and generate some samples.
- Enjoy the generated anime images and videos from your stories!