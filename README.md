# Sentiment Analysis Application for Threat Detection 

A powerful web application that performs sentiment analysis on content from multiple sources including Twitter, Reddit, and news articles. Built with Flask, TensorFlow, and various APIs to provide comprehensive sentiment intelligence.

## Features

- **Multi-source Analysis**: Analyze content from:
  - Twitter users and hashtags
  - Reddit subreddits
  - News articles via NewsAPI
- **Advanced ML Sentiment Analysis**: Utilizes a bidirectional LSTM neural network model for accurate sentiment classification
- **Interactive Web Interface**: Clean Flask-based UI to input search parameters and view analysis results
- **Sentiment Statistics**: View percentage breakdowns of positive, negative, and neutral sentiments
- **Content Grouping**: Results are organized by sentiment category for easy analysis
- 
## How It Works

### Data Collection

- **Twitter**: Uses Selenium to scrape tweets through Nitter instances
- **Reddit**: Uses PRAW (Python Reddit API Wrapper) to fetch posts
- **News**: Uses NewsAPI to collect recent news articles

### Sentiment Analysis

The application employs a sophisticated ML model with the following components:

- **Preprocessing**: Custom text cleaning for social media content
- **Deep Learning Model**: Bidirectional LSTM architecture with embedding layer
- **Classification**: Three-way classification (positive, negative, neutral)
- **Keyword Enhancement**: Additional logic for detecting negative news topics

## Project Structure

- `app.py`: Main Flask application with routes and core functionality
- `models/`: Directory containing trained ML models and preprocessing components
- `templates/`: HTML templates for the web interface
- `twitter_training.csv`: Training data for the sentiment model

## Model Training

The sentiment analysis model is trained on Twitter data with the following architecture:
- Word embedding layer (128 dimensions)
- Bidirectional LSTM layers (64 & 32 units)
- Dense layers with dropout for regularization
- Softmax output for classification

The model automatically trains if no pre-existing model is found, or loads a previously trained model.

## Future Improvements

- Add visualization dashboards for sentiment trends
- Implement user authentication system
- Add historical sentiment tracking
- Support for more languages
- Fine-tune model with domain-specific data
