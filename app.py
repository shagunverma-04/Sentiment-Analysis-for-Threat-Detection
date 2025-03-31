from flask import Flask, render_template, request, flash, jsonify
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import praw
from newsapi import NewsApiClient
import datetime

#Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv()

# Initialize Reddit API client
try: 
    reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)
except Exception as e:
    print(f"Error initializing Reddit API: {e}")
    reddit = None

def test_reddit_connection():
    try:
        # Try to fetch a popular subreddit
        subreddit = reddit.subreddit('python')
        print(f"Successfully connected to Reddit")
        
        # Try to fetch one post to verify permissions
        for post in subreddit.hot(limit=1):
            print(f"Successfully fetched post: {post.title}")
            return True
            
    except Exception as e:
        print(f"Reddit connection test failed: {str(e)}")
        return False

# Add this to your route for testing
@app.route('/test_reddit')
def test_reddit():
    if test_reddit_connection():
        return "Reddit connection successful!"
    return "Reddit connection failed. Check console for details."

def fetch_reddit_posts(subreddit_name, limit=10):
    """Fetch posts from a specified subreddit"""
    try:
        print(f"Attempting to fetch posts from r/{subreddit_name}")  # Debug print
        
        posts = []
        subreddit = reddit.subreddit(subreddit_name)
        
        # Fetch hot posts from the subreddit
        for post in subreddit.hot(limit=limit):
            # Get both title and body text
            title = post.title
            body = post.selftext if hasattr(post, 'selftext') else ''
            text = f"{title} {body}".strip()
            
            if text:  # Only add if there's text content
                posts.append({
                    'text': text,
                    'url': f"https://reddit.com{post.permalink}",
                    'score': post.score,
                    'source': f"Reddit - r/{subreddit_name}"
                })
                print(f"Added post: {title[:50]}...")  # Debug print
        
        print(f"Successfully fetched {len(posts)} posts from r/{subreddit_name}")
        return posts
    
    except Exception as e:
        print(f"Error in fetch_reddit_posts: {str(e)}")
        return []


# Initialize News API client
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

def fetch_reddit_posts(subreddit_name, limit=10):
    """Fetch posts from a specified subreddit"""
    try:
        posts = []
        subreddit = reddit.subreddit(subreddit_name)
        
        for post in subreddit.hot(limit=limit):
            # Combine title and selftext for better context
            text = f"{post.title} {post.selftext}"
            if text.strip():  # Only add if there's text content
                posts.append({
                    'text': text,
                    'url': f"https://reddit.com{post.permalink}",
                    'score': post.score
                })
        
        print(f"Found {len(posts)} Reddit posts")
        return posts
    
    except Exception as e:
        print(f"Error fetching Reddit posts: {str(e)}")
        return []

def fetch_news_articles(query, limit=10):
    """Fetch news articles related to a query"""
    try:
        # Get articles from the last 30 days
        from_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        
        response = newsapi.get_everything(
            q=query,
            language='en',
            from_param=from_date,
            sort_by='relevancy',
            page_size=limit
        )
        
        articles = []
        for article in response['articles']:
            # Combine title and description for analysis
            text = f"{article['title']} {article['description'] or ''}"
            if text.strip():  # Only add if there's text content
                articles.append({
                    'text': text,
                    'url': article['url'],
                    'source': article['source']['name']
                })
        
        print(f"Found {len(articles)} news articles")
        return articles
    
    except Exception as e:
        print(f"Error fetching news articles: {str(e)}")
        return []

#core machine learning component
class TweetSentimentAnalyzer:
    def __init__(self, model_path='models/advanced_sentiment_model.h5', dataset_path='twitter_training.csv'):
        self.model_path = model_path         #stores model and dataset path as instance variables.
        self.dataset_path = dataset_path
        self.model = None                    #initialize
        self.tokenizer = None
        self.label_encoder = LabelEncoder()  #creates an instance of labelencoder that will be used to encode text labels into numrical format for training 
        self.max_length = 100    #max sequence length for text input 
        
        # Paths for saving tokenizer and label encoder
        self.tokenizer_path = 'models/tokenizer.pickle'
        self.label_encoder_path = 'models/label_encoder.pickle'
        
        os.makedirs('models', exist_ok=True) #ensure models directory exists
        
        # Initialize model
        self.load_or_train_model()

    #cleans up tweet -lowercase, remove urls , @mentions , hashtags, special_char
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        # Less aggressive preprocessing - keep more sentiment indicators
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        # Keep hashtags as they may contain sentiment information
        # text = re.sub(r'#\w+', '', text)  # Commented out
        # Keep more punctuation that may indicate sentiment (!, ?, etc.)
        text = re.sub(r'[^\w\s\u263a-\U0001f645!?.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save_tokenizer_and_encoder(self):
        """Save tokenizer and label encoder for future use"""
        try:
            with open(self.tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(self.label_encoder_path, 'wb') as handle:
                pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Tokenizer and label encoder saved successfully")
        except Exception as e:
            print(f"Error saving tokenizer or label encoder: {e}")

    def load_tokenizer_and_encoder(self):
        """Load tokenizer and label encoder from files"""

        try:
            if os.path.exists(self.tokenizer_path):
                with open(self.tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                print("Tokenizer loaded successfully")
            
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as handle:
                    self.label_encoder = pickle.load(handle)
                print("Label encoder loaded successfully")
                print(f"Classes: {self.label_encoder.classes_}")
            
            return self.tokenizer is not None and self.label_encoder is not None
        except Exception as e:
            print(f"Error loading tokenizer or label encoder: {e}")
            return False

    def load_or_train_model(self):
        #checks if there is a pre-existing model
        try:
            model_exists = os.path.exists(self.model_path)
            encoders_exist = self.load_tokenizer_and_encoder()
            
            if model_exists and encoders_exist:
                print("Loading existing model and preprocessing components...")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully")
                # Verify model output shape matches label encoder classes
                print(f"Model output shape: {self.model.output_shape}")
                print(f"Label encoder classes: {self.label_encoder.classes_}")
                return
                
            print("Training new model...")
            self.train_comprehensive_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            self.train_comprehensive_model()

    def train_comprehensive_model(self):
        """Train model with fixed input size and improved architecture"""
        try:
            # Load and prepare data
            df = pd.read_csv(self.dataset_path, encoding='utf-8', on_bad_lines='skip')
            df.columns = ["ID", "Category", "Sentiment", "Tweet"]
            df = df[["Sentiment", "Tweet"]].dropna()
            
            # Print data statistics for debugging
            print(f"Dataset shape: {df.shape}")
            print(f"Sentiment distribution:\n{df['Sentiment'].value_counts()}")
            
            # Map all sentiment labels to lowercase standard format 
            sentiment_mapping = {
                'Positive': 'positive',
                'Negative': 'negative',
                'Neutral': 'neutral',
                'Irrelevant': 'neutral'  # Map irrelevant to neutral
            }
            
            df['Sentiment'] = df['Sentiment'].replace(sentiment_mapping)
            print(f"Updated sentiment distribution after mapping:\n{df['Sentiment'].value_counts()}")
            
            # Preprocess texts
            texts = df['Tweet'].apply(self.preprocess_text).tolist()
            labels = df['Sentiment'].tolist()
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
            encoded_labels = self.label_encoder.transform(labels)
            categorical_labels = tf.keras.utils.to_categorical(encoded_labels)
            
            print(f"Label encoder classes: {self.label_encoder.classes_}")
            
            # Convert words into numerical tokens
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=5000, 
                oov_token="<OOV>"
            )
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            
            # Since LSTM requires fixed size inputs, max-length is set
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, 
                maxlen=self.max_length,
                padding='post',
                truncating='post'
            )
            
            # Split data - without stratification since we've fixed the classes
            X_train, X_test, y_train, y_test = train_test_split(
                padded_sequences,
                categorical_labels,
                test_size=0.2,
                random_state=42
                # Removed stratify parameter
            )
            
            # Build model with fixed input shape
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32),
                tf.keras.layers.Embedding(5000, 128, input_length=self.max_length),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
            ])

            # Compile and train
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Add early stopping and reduce learning rate on plateau
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
            
            # Train the model
            history = self.model.fit(
                X_train,
                y_train,
                epochs=15,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            evaluation = self.model.evaluate(X_test, y_test)
            print(f"Test accuracy: {evaluation[1]:.4f}")
            
            # Save model, tokenizer, and label encoder
            self.model.save(self.model_path)
            self.save_tokenizer_and_encoder()
            print("Model, tokenizer, and label encoder saved successfully")
            
        except Exception as e:
            print(f"Error during model training: {e}")
            raise

    def predict_sentiment(self, text):
        """Predict sentiment with enhanced negative news detection"""
        try:    
            # List of words/phrases that indicate negative news
            negative_keywords = [
                'attack', 'kill', 'death', 'dead', 'hostage', 'violence', 'terrorist',
                'murder', 'casualty', 'casualties', 'wounded', 'injured', 'victim',
                'tragedy', 'tragic', 'war', 'conflict', 'disaster', 'emergency',
                'crisis', 'fatal', 'died', 'killed', 'shooting', 'shot', 'massacre',
                'bombing', 'explosion', 'terrorist', 'terrorism', 'assault'
            ]
            
            # First check for negative keywords
            text_lower = text.lower()
            for keyword in negative_keywords:
                if keyword in text_lower:
                    print(f"Negative keyword found: {keyword}")
                    return {'sentiment': 'negative', 'confidence': 0.95}
            
            # If no negative keywords, proceed with model prediction
            if not self.model or not self.tokenizer:
                print("Model or tokenizer not loaded.")
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            processed_text = self.preprocess_text(text)
            print(f"Processed text: {processed_text[:100]}...")
            
            if not processed_text:
                print("Empty text after preprocessing")
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=self.max_length, padding='post', truncating='post'
            )
            
            prediction = self.model.predict(padded_sequence, verbose=0)[0]
            predicted_class_index = np.argmax(prediction)
            
            # Ensure the index is valid
            if predicted_class_index < len(self.label_encoder.classes_):
                predicted_class = self.label_encoder.classes_[predicted_class_index]
            else:
                print(f"Warning: Index {predicted_class_index} out of bounds")
                predicted_class = "neutral"
            
            confidence = float(np.max(prediction))
            
            print(f"Final prediction: {predicted_class} with confidence {confidence:.4f}")
            
            return {'sentiment': predicted_class, 'confidence': confidence}
        
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}

def fetch_tweets(query, limit=10, retries=3):
    """Enhanced tweet fetching with authentication handling"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    # Add these new options to better handle modern Twitter
    options.add_argument("--enable-javascript")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    tweets = []
    
    print(f"Starting tweet fetch for query: {query}")

    for attempt in range(retries):
        driver = None
        try:
            print(f"Attempt {attempt + 1} of {retries}")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            # Update the stealth settings
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Try Nitter as an alternative to Twitter
            nitter_instances = [
                "https://nitter.net",
                "https://nitter.lacontrevoie.fr",
                "https://nitter.1d4.us"
            ]
            
            for instance in nitter_instances:
                try:
                    search_url = f"{instance}/search?f=tweets&q={query}"
                    print(f"Trying Nitter instance: {search_url}")
                    
                    driver.get(search_url)
                    time.sleep(5)  # Wait for page load
                    
                    # Nitter specific selectors
                    tweet_elements = driver.find_elements(By.CSS_SELECTOR, ".timeline-item")
                    
                    if tweet_elements:
                        print(f"Found {len(tweet_elements)} tweets on {instance}")
                        
                        for tweet in tweet_elements[:limit]:
                            try:
                                tweet_text = tweet.find_element(By.CSS_SELECTOR, ".tweet-content").text
                                if tweet_text and tweet_text not in [t['text'] for t in tweets]:
                                    tweets.append({'text': tweet_text})
                                    print(f"Found tweet: {tweet_text[:50]}...")
                            except Exception as e:
                                print(f"Error extracting tweet text: {str(e)}")
                                continue
                        
                        if tweets:
                            break  # Found tweets, exit instance loop
                            
                except Exception as e:
                    print(f"Error with Nitter instance {instance}: {str(e)}")
                    continue
            
            if tweets:
                break  # Exit retry loop if we have tweets
                
        except Exception as e:
            print(f"Error during tweet fetching: {str(e)}")
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            print(f"Attempt {attempt + 1} completed")
    
    if not tweets:
        print("Failed to fetch any tweets after all attempts")
    
    return tweets[:limit]

# Flask application setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize sentiment analyzer
sentiment_analyzer = TweetSentimentAnalyzer()

def calculate_sentiment_stats(analyzed_tweets):
    """
    Calculate sentiment statistics from analyzed tweets
    
    Args:
        analyzed_tweets (list): List of dictionaries containing analyzed tweets
        
    Returns:
        dict: Dictionary containing percentages for each sentiment category
    """
    try:
        # Get list of all sentiments
        ml_sentiments = [t['ml_sentiment'] for t in analyzed_tweets]
        total = len(ml_sentiments)
        
        if total == 0:
            return {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # Calculate percentages for each sentiment
        stats = {
            'positive': round(ml_sentiments.count('positive') / total * 100, 1),
            'negative': round(ml_sentiments.count('negative') / total * 100, 1),
            'neutral': round(ml_sentiments.count('neutral') / total * 100, 1)
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating sentiment stats: {str(e)}")
        return {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        return render_template('sentiment.html')
    
    try:
        # Get form inputs
        userid = request.form.get('userid', '').strip()
        hashtag = request.form.get('hashtag', '').strip()
        subreddit = request.form.get('subreddit', '').strip()
        news_query = request.form.get('news_query', '').strip()
        
        # Initialize variables
        analyzed_content = []
        stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        source_type = ""
        
        # Check if any input is provided
        if not any([userid, hashtag, subreddit, news_query]):
            flash("Please enter at least one search term", "error")
            return render_template('sentiment.html')
        
        # Handle Reddit analysis
        if subreddit:
            try:
                print(f"Fetching posts from r/{subreddit}")
                posts = fetch_reddit_posts(subreddit, limit=20)
                source_type = "reddit posts"
                
                if not posts:
                    flash(f"No posts found in r/{subreddit}. The subreddit might be private or empty.", "error")
                    return render_template('sentiment.html')
                
                for post in posts:
                    try:
                        ml_sentiment = sentiment_analyzer.predict_sentiment(post['text'])
                        analyzed_content.append({
                            'text': post['text'][:300] + '...' if len(post['text']) > 300 else post['text'],
                            'ml_sentiment': ml_sentiment['sentiment'],
                            'ml_confidence': round(float(ml_sentiment['confidence']), 3),
                            'url': post['url'],
                            'score': post['score'],
                            'source': f"Reddit - r/{subreddit}"
                        })
                    except Exception as e:
                        print(f"Error analyzing post: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error processing subreddit: {str(e)}")
                flash(f"Error accessing r/{subreddit}: {str(e)}", "error")
                return render_template('sentiment.html')
        
        # Handle Twitter analysis
        elif userid or hashtag:
            try:
                query = f"from:{userid}" if userid else f"#{hashtag}"
                tweets = fetch_tweets(query, limit=20)
                source_type = "tweets"
                
                if not tweets:
                    flash("No tweets found. Please try again.", "error")
                    return render_template('sentiment.html')
                
                for tweet in tweets:
                    try:
                        ml_sentiment = sentiment_analyzer.predict_sentiment(tweet['text'])
                        analyzed_content.append({
                            'text': tweet['text'],
                            'ml_sentiment': ml_sentiment['sentiment'],
                            'ml_confidence': round(float(ml_sentiment['confidence']), 3),
                            'source': 'Twitter'
                        })
                    except Exception as e:
                        print(f"Error analyzing tweet: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error processing tweets: {str(e)}")
                flash(f"Error accessing Twitter: {str(e)}", "error")
                return render_template('sentiment.html')
        
        # Handle News analysis
        elif news_query:
            try:
                articles = fetch_news_articles(news_query, limit=20)
                source_type = "news articles"
                
                if not articles:
                    flash("No news articles found. Please try again.", "error")
                    return render_template('sentiment.html')
                
                for article in articles:
                    try:
                        ml_sentiment = sentiment_analyzer.predict_sentiment(article['text'])
                        analyzed_content.append({
                            'text': article['text'][:300] + '...' if len(article['text']) > 300 else article['text'],
                            'ml_sentiment': ml_sentiment['sentiment'],
                            'ml_confidence': round(float(ml_sentiment['confidence']), 3),
                            'url': article['url'],
                            'source': f"News - {article['source']}"
                        })
                    except Exception as e:
                        print(f"Error analyzing article: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error processing news: {str(e)}")
                flash(f"Error accessing news: {str(e)}", "error")
                return render_template('sentiment.html')
        
        # Check if any content was successfully analyzed
        if not analyzed_content:
            flash("Could not analyze any content. Please try again.", "error")
            return render_template('sentiment.html')
        
        # Calculate statistics
        stats = calculate_sentiment_stats(analyzed_content)
        
        # Group content by sentiment
        grouped_content = {
            'positive': [x for x in analyzed_content if x['ml_sentiment'] == 'positive'],
            'negative': [x for x in analyzed_content if x['ml_sentiment'] == 'negative'],
            'neutral': [x for x in analyzed_content if x['ml_sentiment'] == 'neutral']
        }
        
        # Return the template with all data
        return render_template(
            'sentiment.html',
            positive=stats['positive'],
            negative=stats['negative'],
            neutral=stats['neutral'],
            analyzed_content=analyzed_content,
            grouped_content=grouped_content,
            source_type=source_type
        )
    
    except Exception as e:
        print(f"Route error: {str(e)}")
        flash(f"An error occurred: {str(e)}", "error")
        return render_template('sentiment.html')


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)