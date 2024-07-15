import requests
from bs4 import BeautifulSoup

# Example function to scrape data from a fashion blog
def scrape_fashion_blog(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract relevant data
    articles = soup.find_all('article')
    for article in articles:
        title = article.find('h2').text
        summary = article.find('p').text
        print(f"Title: {title}\nSummary: {summary}\n")

# Example URL
scrape_fashion_blog('https://examplefashionblog.com')

## DATA STORAGE
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('fashion.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS articles (title TEXT, summary TEXT)''')

# Insert data
c.execute("INSERT INTO articles (title, summary) VALUES (?, ?)", (title, summary))
conn.commit()
conn.close()

## DATA PROCESSING
import pandas as pd

# Extract data from database
conn = sqlite3.connect('fashion.db')
df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()

# Transform data (example: clean text)
df['clean_summary'] = df['summary'].apply(lambda x: x.lower().strip())

# Load data into another database or process further
df.to_csv('cleaned_data.csv', index=False)

## TREND ANALYSIS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load cleaned data
df = pd.read_csv('cleaned_data.csv')

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_summary'])

# Perform LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)

# Display topics
for index, topic in enumerate(lda.components_):
    print(f"Topic {index}:")
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])

## RECCOMENDATION ENGINE
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load example data (user-item interactions)
data = Dataset.load_builtin('ml-100k')  # Example dataset

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25)

# Use SVD for collaborative filtering
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

# Evaluate accuracy
accuracy.rmse(predictions)

# Example: Get recommendations for a specific user
user_id = str(196)  # Example user ID
items = [i for i in range(1, 1683)]  # Example item IDs
predictions = [algo.predict(user_id, str(i)) for i in items]
recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
print("Top recommendations:")
for rec in recommendations:
    print(rec.iid, rec.est)

## USER INTERFACE
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = request.form['user_id']
    
    # Generate recommendations (mock example)
    recommendations = generate_recommendations(user_id)
    
    return render_template('recommendations.html', recommendations=recommendations)

def generate_recommendations(user_id):
    # Placeholder function for generating recommendations
    return ["Item 1", "Item 2", "Item 3"]

if __name__ == '__main__':
    app.run(debug=True)
