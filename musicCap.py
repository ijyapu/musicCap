import pandas as pd
import re
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Load the CSV file
data = pd.read_csv("musiccaps-public.csv")

# Data Cleaning
data = data.drop_duplicates(subset=['caption']).dropna(subset=['caption'])

# Initialize Porter Stemmer and Stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing: Tokenize, clean text, remove stopwords, and stem
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['cleaned_caption'] = data['caption'].apply(clean_text)

# Exploratory Data Analysis: Word Cloud Visualization
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in MusicCap Dataset')
    plt.show()

# Generate a word cloud with cleaned captions
all_text = ' '.join(data['cleaned_caption'])
generate_word_cloud(all_text)

# Analyze Data with TF-IDF and PCA for Visualization
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X_tfidf = vectorizer.fit_transform(data['cleaned_caption']).toarray()

# Reduce dimensions for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf)

# Plot the PCA-reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title('Caption Data PCA Plot')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Generate random captions from the cleaned dataset
def generate_random_caption():
    return random.choice(data['cleaned_caption'].values)

# User interaction for generating random captions
print("Welcome to the MusicCap Caption Generator!")
while True:
    user_input = input("\nPress Enter to generate a random music-related caption ('exit' to quit): ").strip().lower()
    if user_input == 'exit':
        break
    print("Random caption:", generate_random_caption())

print("\nThank you for exploring the MusicCap dataset!")
