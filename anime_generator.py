# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import zipfile
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Reshape, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, Bleu, Rouge, Meteor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Define some constants
DATA_DIR = 'data' # The directory to store the data
SCRIPT_URL = 'https://www.animenewsnetwork.com/encyclopedia/api.xml?title=~' # The base URL to get the script of an anime title
SUBTITLE_URL = 'https://animetosho.org/search?search_type=1&term=' # The base URL to get the subtitle of an anime title
IMAGE_URL = 'https://cdn.myanimelist.net/images/anime/' # The base URL to get the image of an anime title
MAX_SCRIPTS = 1000 # The maximum number of scripts to download
MAX_SUBTITLES = 1000 # The maximum number of subtitles to download
MAX_IMAGES = 1000 # The maximum number of images to download

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

# Define a function to download a file from a URL
def download_file(url, filename):
  # Send a GET request to the URL
  response = requests.get(url)
  # Check if the response is successful
  if response.status_code == 200:
    # Write the content of the response to a file
    with open(filename, 'wb') as f:
      f.write(response.content)
    # Return True if the file is downloaded successfully
    return True
  else:
    # Return False if the response is not successful
    return False

# Define a function to unzip a file
def unzip_file(filename, dirname):
  # Open the zip file
  with zipfile.ZipFile(filename, 'r') as zf:
    # Extract all the files to the directory
    zf.extractall(dirname)

# Define a function to get the script of an anime title
def get_script(title):
  # Encode the title to URL format
  title = title.replace(' ', '+')
  # Construct the URL to get the script
  url = SCRIPT_URL + title
  # Download the XML file
  filename = os.path.join(DATA_DIR, title + '.xml')
  success = download_file(url, filename)
  # Check if the file is downloaded successfully
  if success:
    # Parse the XML file
    with open(filename, 'r') as f:
      xml = f.read()
    # Find the script tag
    match = re.search(r'<script>(.*?)</script>', xml, re.DOTALL)
    # Check if the script tag exists
    if match:
      # Get the script content
      script = match.group(1)
      # Return the script
      return script
    else:
      # Return None if the script tag does not exist
      return None
  else:
    # Return None if the file is not downloaded successfully
    return None

# Define a function to get the subtitle of an anime title
def get_subtitle(title):
  # Encode the title to URL format
  title = title.replace(' ', '+')
  # Construct the URL to get the subtitle
  url = SUBTITLE_URL + title
  # Send a GET request to the URL
  response = requests.get(url)
  # Check if the response is successful
  if response.status_code == 200:
    # Get the HTML content of the response
    html = response.text
    # Find the first link to a subtitle file
    match = re.search(r'<a href="(.*?\.zip)"', html)
    # Check if the link exists
    if match:
      # Get the link
      link = match.group(1)
      # Download the zip file
      filename = os.path.join(DATA_DIR, title + '.zip')
      success = download_file(link, filename)
      # Check if the file is downloaded successfully
      if success:
        # Unzip the file
        dirname = os.path.join(DATA_DIR, title)
        unzip_file(filename, dirname)
        # Find the first subtitle file in the directory
        for file in os.listdir(dirname):
          if file.endswith('.srt'):
            # Get the subtitle file name
            subtitle = os.path.join(dirname, file)
            # Return the subtitle file name
            return subtitle
        # Return None if no subtitle file is found
        return None
      else:
        # Return None if the file is not downloaded successfully
        return None
    else:
      # Return None if the link does not exist
      return None
  else:
    # Return None if the response is not successful
    return None

# Define a function to get the image of an anime title
def get_image(title):
  # Encode the title to URL format
  title = title.replace(' ', '+')
  # Construct the URL to get the image
  url = IMAGE_URL + title + '.jpg'
  # Download the image file
  filename = os.path.join(DATA_DIR, title + '.jpg')
  success = download_file(url, filename)
  # Check if the file is downloaded successfully
  if success:
    # Return the image file name
    return filename
  else:
    # Return None if the file is not downloaded successfully
    return None

# Define a list of anime titles to download
anime_titles = ['Naruto', 'One Piece', 'Dragon Ball Super', 'Attack on Titan', 'My Hero Academia', 'Demon Slayer', 'Death Note', 'Fullmetal Alchemist', 'Code Geass', 'Steins;Gate']

# Initialize some counters
script_count = 0
subtitle_count = 0
image_count = 0

# Loop through the anime titles
for title in anime_titles:
  # Print the current title
  print('Downloading data for', title)
  # Get the script of the title
  script = get_script(title)
  # Check if the script exists
  if script:
    # Increment the script counter
    script_count += 1
    # Print the script
    print('Script:', script)
  else:
    # Print a message if the script does not exist
    print('No script found for', title)
  # Get the subtitle of the title
  subtitle = get_subtitle(title)
  # Check if the subtitle exists
  if subtitle:
    # Increment the subtitle counter
    subtitle_count += 1
    # Print the subtitle
    print('Subtitle:', subtitle)
  else:
    # Print a message if the subtitle does not exist
    print('No subtitle found for', title)
  # Get the image of the title
  image = get_image(title)
  # Check if the image exists
  if image:
    # Increment the image counter
    image_count += 1
    # Print the image
    print('Image:', image)
  else:
    # Print a message if the image does not exist
    print('No image found for', title)
  # Print a separator
  print('-' * 80)

# Print the summary of the data collection
print('Data collection summary:')
print('Scripts downloaded:', script_count, '/', MAX_SCRIPTS)
print('Subtitles downloaded:', subtitle_count, '/', MAX_SUBTITLES)
print('Images downloaded:', image_count, '/', MAX_IMAGES)
