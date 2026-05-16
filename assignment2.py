import PyPDF2
import re
import string
import nltk
import pandas as pd
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK Data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================
# READ PDF
# ==========================

pdf_file = open('alice.pdf', 'rb')

pdf_reader = PyPDF2.PdfReader(pdf_file)

total_pages = len(pdf_reader.pages)

print("\n===== TOTAL PAGES =====")
print(total_pages)

# ==========================
# EXTRACT TEXT
# ==========================

text = ""

for page in pdf_reader.pages:
    text += page.extract_text()

print("\n===== SAMPLE EXTRACTED TEXT =====")
print(text[:1000])

# ==========================
# PREPROCESSING
# ==========================

# Convert to lowercase
text = text.lower()

# Remove numbers
text = re.sub(r'\d+', '', text)

# Remove special symbols
text = re.sub(r'[^a-zA-Z\s]', '', text)

# Remove extra spaces
text = re.sub(r'\s+', ' ', text)

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

print("\n===== CLEANED TEXT =====")
print(text[:500])

# ==========================
# TOKENIZATION
# ==========================

words = text.split()

# ==========================
# STOP WORD REMOVAL
# ==========================

stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words if word not in stop_words]

stopword_count = len(words) - len(filtered_words)

print("\n===== STOP WORD COUNT =====")
print(stopword_count)

print("\n===== VALID WORD COUNT =====")
print(len(filtered_words))

# ==========================
# STEMMING
# ==========================

ps = PorterStemmer()

stemmed_words = [ps.stem(word) for word in filtered_words[:50]]

print("\n===== STEMMING OUTPUT =====")
print(stemmed_words)

# ==========================
# LEMMATIZATION
# ==========================

lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words[:50]]

print("\n===== LEMMATIZATION OUTPUT =====")
print(lemmatized_words)

# ==========================
# ONE HOT ENCODING
# ==========================

sample_text = [' '.join(filtered_words[:20])]

cv = CountVectorizer(binary=True)

one_hot = cv.fit_transform(sample_text)

df_onehot = pd.DataFrame(
    one_hot.toarray(),
    columns=cv.get_feature_names_out()
)

print("\n===== ONE HOT ENCODING =====")
print(df_onehot)

# ==========================
# TF-IDF
# ==========================

clean_text = ' '.join(filtered_words)

tfidf = TfidfVectorizer(max_features=20)

tfidf_matrix = tfidf.fit_transform([clean_text])

feature_names = tfidf.get_feature_names_out()

tfidf_values = tfidf_matrix.toarray()[0]

df_tfidf = pd.DataFrame({
    'Word': feature_names,
    'TF-IDF Score': tfidf_values
})

print("\n===== TF-IDF OUTPUT =====")
print(df_tfidf)

# ==========================
# PLOTLY GRAPH
# ==========================

fig = px.scatter(
    df_tfidf,
    x='Word',
    y='TF-IDF Score',
    title='TF-IDF Scatter Plot' 
)

fig.show()