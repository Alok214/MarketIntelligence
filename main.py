"""
Market Intelligence Data Collection and Analysis System
======================================================

A comprehensive system for collecting and analyzing real-time market intelligence
from Twitter/X discussions about Indian stock markets.

Author: Market Intelligence System
Date: 30 August 2025
"""

import os
import sys
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import pickle

# Data processing and analysis
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Twitter API
import tweepy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration and utilities
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_intelligence.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TweetData:
    """Data structure for storing tweet information."""
    tweet_id: str
    username: str
    timestamp: datetime
    content: str
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    mentions: List[str]
    hashtags: List[str]
    url: str
    language: str
    verified: bool
    followers_count: int
    following_count: int
    account_created: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['account_created'] = self.account_created.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TweetData':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['account_created'] = datetime.fromisoformat(data['account_created'])
        return cls(**data)

class TwitterCollector:
    """Twitter data collector with rate limiting and error handling."""
    
    def __init__(self):
        self.api = self._setup_twitter_api()
        self.rate_limit_tracker = defaultdict(list)
        self.collected_tweets = deque(maxlen=100)
        self.tweet_ids_seen = set()
        self.collection_stats = {
            'total_collected': 0,
            'duplicates_filtered': 0,
            'api_calls_made': 0,
            'rate_limits_hit': 0
        }
        self.target_hashtags = [
            '#sensex'
        ]
        # '#indianmarket', '#equity', '#options', '#futures', ','#banknifty', '#nifty50', 
        # '#nse', '#bse', '#stockmarket', '#trading', , '#intraday'
    
    def _setup_twitter_api(self) -> tweepy.Client:
        """Setup Twitter API v2 with credentials from environment."""
        try:
            client = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                consumer_key=os.getenv('TWITTER_API_KEY'),
                consumer_secret=os.getenv('TWITTER_API_SECRET'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
                wait_on_rate_limit=True
            )
            user = client.get_me().data
            logger.info(f"Twitter API v2 authentication successful as {user.username}")
            return client
        except Exception as e:
            logger.error(f"Twitter API v2 setup failed: {e}")
            raise
    
    
    def _is_rate_limited(self, endpoint: str) -> bool:
        """Check if we're rate limited for a specific endpoint."""
        now = time.time()
        calls = self.rate_limit_tracker[endpoint]
        calls = [call_time for call_time in calls if now - call_time < 900]
        self.rate_limit_tracker[endpoint] = calls
        return len(calls) >= 180
    
    def _track_api_call(self, endpoint: str):
        """Track API call for rate limiting."""
        self.rate_limit_tracker[endpoint].append(time.time())
        self.collection_stats['api_calls_made'] += 1
    
    def _extract_tweet_data(self, tweet, user) -> Optional[TweetData]:
        """Extract relevant data from tweet object."""
        try:
            entities = tweet.entities or {}
            mentions = [mention['username'] for mention in entities.get('mentions', [])]
            hashtags = [hashtag['tag'].lower() for hashtag in entities.get('hashtags', [])]
            
            tweet_data = TweetData(
                tweet_id=str(tweet.id),
                username=user.username if user else 'unknown',
                timestamp=tweet.created_at,
                content=tweet.text,
                retweet_count=tweet.public_metrics.get('retweet_count', 0),
                like_count=tweet.public_metrics.get('like_count', 0),
                reply_count=tweet.public_metrics.get('reply_count', 0),
                quote_count=tweet.public_metrics.get('quote_count', 0),
                mentions=mentions,
                hashtags=hashtags,
                url=f"https://twitter.com/{user.username if user else 'user'}/status/{tweet.id}",
                language=tweet.lang,
                verified=user.verified if user else False,
                followers_count=user.public_metrics.get('followers_count', 0) if user else 0,
                following_count=user.public_metrics.get('following_count', 0) if user else 0,
                account_created=user.created_at if user else tweet.created_at
            )
            return tweet_data
        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return None
    
    def _is_relevant_tweet(self, tweet_data: TweetData) -> bool:
        """Check if tweet is relevant to Indian stock market."""
        content_lower = tweet_data.content.lower()
        hashtags_lower = [h.lower() for h in tweet_data.hashtags]
        
        target_found = any(hashtag in hashtags_lower for hashtag in 
                          [h.replace('#', '') for h in self.target_hashtags])
        
        market_keywords = [
            'nifty', 'sensex', 'nse', 'bse', 'stock', 'trading',
            'buy', 'sell', 'target', 'stoploss', 'breakout',
            'support', 'resistance', 'bullish', 'bearish',
            'rupee', 'inr', 'market', 'equity', 'options'
        ]
        keyword_found = any(keyword in content_lower for keyword in market_keywords)
        
        indian_terms = [
            'reliance', 'hdfc', 'icici', 'sbi', 'tcs',
            'infosys', 'bajaj', 'maruti', 'asian paints',
            'kotak', 'axis', 'wipro', 'larsen'
        ]
        indian_found = any(term in content_lower for term in indian_terms)
        
        return target_found or (keyword_found and (indian_found or 'india' in content_lower))
    
    def collect_tweets_by_hashtag(self, hashtag: str, count: int = 100) -> List[TweetData]:
        """Collect tweets for a specific hashtag using Twitter API v2."""
        if self._is_rate_limited('search'):
            logger.warning(f"Rate limited for search endpoint")
            time.sleep(60)

        tweets_collected = []
        try:
            self._track_api_call('search')
            query = f"{hashtag} lang:en -is:retweet"
            response = self.api.search_recent_tweets(
                query=query,
                max_results=min(count, 100),
                tweet_fields=['created_at', 'public_metrics', 'entities', 'lang'],
                user_fields=['username', 'verified', 'public_metrics', 'created_at'],
                expansions=['author_id']
            )

            if response.data:
                users = {user.id: user for user in response.includes.get('users', [])}
                for tweet in response.data:
                    if str(tweet.id) in self.tweet_ids_seen:
                        self.collection_stats['duplicates_filtered'] += 1
                        continue

                    user = users.get(tweet.author_id)
                    tweet_data = self._extract_tweet_data(tweet, user)
                    if tweet_data and self._is_relevant_tweet(tweet_data):
                        if tweet_data.timestamp > datetime.now() - timedelta(days=1):
                            tweets_collected.append(tweet_data)
                            self.tweet_ids_seen.add(tweet_data.tweet_id)
                            self.collected_tweets.append(tweet_data)
                            self.collection_stats['total_collected'] += 1
                            logger.info(f"Collected tweet from @{tweet_data.username}: {tweet_data.content[:50]}...")
            else:
                logger.warning(f"No tweets found for hashtag {hashtag}")

        except tweepy.TooManyRequests:
            logger.warning(f"Rate limit exceeded for hashtag {hashtag}")
            self.collection_stats['rate_limits_hit'] += 1
            time.sleep(900)
        except tweepy.TweepyException as e:
            logger.error(f"Error collecting tweets for {hashtag}: {e}")
            if '403' in str(e):
                logger.error("Check your API access level. Search endpoint may not be available.")
        except Exception as e:
            logger.error(f"Unexpected error for {hashtag}: {e}")

        return tweets_collected
    
    def collect_market_tweets(self, target_count: int = 2000) -> List[TweetData]:
        """Collect tweets about Indian stock market."""
        all_tweets = []
        tweets_per_hashtag = max(10, target_count // len(self.target_hashtags))  # Ensure at least 10 tweets per hashtag
        max_attempts = 3  # Retry attempts for rate limits
        
        logger.info(f"Starting tweet collection. Target: {target_count} tweets")
        
        for hashtag in self.target_hashtags:
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Collecting tweets for {hashtag}, attempt {attempt + 1}")
                    tweets = self.collect_tweets_by_hashtag(hashtag, tweets_per_hashtag)
                    all_tweets.extend(tweets)
                    logger.info(f"Collected {len(tweets)} tweets for {hashtag}")
                    time.sleep(10)  # Delay between hashtags to avoid rate limits
                    break  # Success, move to next hashtag
                except tweepy.TooManyRequests:
                    logger.warning(f"Rate limit exceeded for {hashtag}. Waiting {900 * (2 ** attempt)} seconds")
                    self.collection_stats['rate_limits_hit'] += 1
                    time.sleep(900 * (2 ** attempt))  # Exponential backoff: 15min, 30min, 60min
                except tweepy.TweepyException as e:
                    logger.error(f"Error collecting tweets for {hashtag}: {e}")
                    break  # Non-rate-limit error, skip to next hashtag
        
        logger.info(f"Collection complete. Total tweets: {len(all_tweets)}")
        if not all_tweets:
            logger.warning("No tweets collected. Check API access, query, or tweet availability.")
        return all_tweets

class DataProcessor:
    """Process and clean collected tweet data."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Indian market specific terms to preserve
        self.market_terms = {
            'nifty', 'sensex', 'banknifty', 'nse', 'bse',
            'bullish', 'bearish', 'breakout', 'support', 'resistance'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle Unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove special characters but keep market symbols
        text = re.sub(r'[^\w\s#@$]', ' ', text)
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from tweet text."""
        cleaned_text = self.clean_text(text)
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@'),
            'url_count': len(re.findall(r'http\S+', text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
        }
        
        # Market sentiment indicators
        bullish_words = ['buy', 'bull', 'bullish', 'long', 'target', 'breakout', 'rally']
        bearish_words = ['sell', 'bear', 'bearish', 'short', 'fall', 'crash', 'dump']
        
        features['bullish_score'] = sum(1 for word in bullish_words if word in cleaned_text.lower())
        features['bearish_score'] = sum(1 for word in bearish_words if word in cleaned_text.lower())
        features['sentiment_ratio'] = features['bullish_score'] - features['bearish_score']
        
        return features
    
    def process_tweets(self, tweets: List[TweetData]) -> pd.DataFrame:
        """Process tweets into structured DataFrame."""
        logger.info(f"Processing {len(tweets)} tweets")
        
        processed_data = []
        
        for tweet in tweets:
            try:
                # Basic tweet data
                row = {
                    'tweet_id': tweet.tweet_id,
                    'username': tweet.username,
                    'timestamp': tweet.timestamp,
                    'content': tweet.content,
                    'cleaned_content': self.clean_text(tweet.content),
                    'retweet_count': tweet.retweet_count,
                    'like_count': tweet.like_count,
                    'reply_count': tweet.reply_count,
                    'quote_count': tweet.quote_count,
                    'total_engagement': (tweet.retweet_count + tweet.like_count + 
                                       tweet.reply_count + tweet.quote_count),
                    'mentions': tweet.mentions,
                    'hashtags': tweet.hashtags,
                    'url': tweet.url,
                    'language': tweet.language,
                    'verified': tweet.verified,
                    'followers_count': tweet.followers_count,
                    'following_count': tweet.following_count,
                    'account_age_days': (tweet.timestamp - tweet.account_created).days
                }
                
                # Extract text features
                features = self.extract_features(tweet.content)
                row.update(features)
                
                processed_data.append(row)
                
            except Exception as e:
                logger.error(f"Error processing tweet {tweet.tweet_id}: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Successfully processed {len(df)} tweets")
        
        return df

class SignalGenerator:
    """Generate trading signals from processed tweet data."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
    def create_text_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """Create TF-IDF vectors from tweet content."""
        logger.info("Creating TF-IDF vectors")
        
        # Fit and transform the text data
        text_vectors = self.tfidf_vectorizer.fit_transform(df['cleaned_content'])
        
        # Convert to dense array and apply PCA for dimensionality reduction
        dense_vectors = text_vectors.toarray()
        reduced_vectors = self.pca.fit_transform(dense_vectors)
        
        logger.info(f"Created {reduced_vectors.shape[1]} dimensional text vectors")
        return reduced_vectors
    
    def calculate_engagement_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate engagement-based signals."""
        engagement_features = [
            'total_engagement', 'retweet_count', 'like_count',
            'followers_count', 'verified', 'hashtag_count'
        ]
        
        # Create engagement score
        df['engagement_score'] = (
            df['like_count'] * 1 +
            df['retweet_count'] * 2 +
            df['reply_count'] * 1.5 +
            df['quote_count'] * 1.5
        ) * (1 + df['verified'].astype(int) * 0.5)
        
        # Normalize by follower count to get engagement rate
        df['engagement_rate'] = df['engagement_score'] / np.maximum(df['followers_count'], 1)
        
        engagement_data = df[engagement_features + ['engagement_score', 'engagement_rate']].fillna(0)
        
        # Scale the features
        scaled_engagement = self.scaler.fit_transform(engagement_data)
        
        return scaled_engagement
    
    def generate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate composite trading signals - this was missing!"""
        logger.info("Generating composite trading signals")
        
        # Create text vectors
        text_vectors = self.create_text_vectors(df)
        
        # Calculate engagement signals
        engagement_signals = self.calculate_engagement_signals(df)
        
        # Generate signals
        df_signals = df.copy()
        
        # Sentiment-based signal
        df_signals['sentiment_signal'] = df_signals['sentiment_ratio'] / np.maximum(
            df_signals['word_count'], 1
        )
        
        # Volume signal (based on tweet frequency)
        df_signals['hour'] = df_signals['timestamp'].dt.hour
        hourly_counts = df_signals.groupby('hour').size()
        df_signals['volume_signal'] = df_signals['hour'].map(hourly_counts)
        
        # Influence signal (based on user metrics)
        df_signals['influence_signal'] = (
            np.log1p(df_signals['followers_count']) * 
            df_signals['engagement_rate'] *
            (1 + df_signals['verified'].astype(int))
        )
        
        # Composite signal
        df_signals['composite_signal'] = (
            df_signals['sentiment_signal'] * 0.4 +
            np.log1p(df_signals['volume_signal']) * 0.3 +
            df_signals['influence_signal'] * 0.3
        )
        
        # Calculate confidence intervals
        signal_std = df_signals['composite_signal'].std()
        df_signals['signal_upper'] = df_signals['composite_signal'] + 1.96 * signal_std
        df_signals['signal_lower'] = df_signals['composite_signal'] - 1.96 * signal_std
        
        logger.info("Signal generation complete")
        return df_signals
    
    def generate_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate composite trading signals."""
        logger.info("Generating composite trading signals")
        
        # Create text vectors
        text_vectors = self.create_text_vectors(df)
        
        # Calculate engagement signals
        engagement_signals = self.calculate_engagement_signals(df)
        
        # Combine signals
        combined_features = np.hstack([text_vectors, engagement_signals])
        
        # Generate signals
        df_signals = df.copy()
        
        # Sentiment-based signal
        df_signals['sentiment_signal'] = df_signals['sentiment_ratio'] / np.maximum(
            df_signals['word_count'], 1
        )
        
        # Volume signal (based on tweet frequency)
        df_signals['hour'] = df_signals['timestamp'].dt.hour
        hourly_counts = df_signals.groupby('hour').size()
        df_signals['volume_signal'] = df_signals['hour'].map(hourly_counts)
        
        # Influence signal (based on user metrics)
        df_signals['influence_signal'] = (
            np.log1p(df_signals['followers_count']) * 
            df_signals['engagement_rate'] *
            (1 + df_signals['verified'].astype(int))
        )
        
        # Composite signal
        df_signals['composite_signal'] = (
            df_signals['sentiment_signal'] * 0.4 +
            np.log1p(df_signals['volume_signal']) * 0.3 +
            df_signals['influence_signal'] * 0.3
        )
        
        # Calculate confidence intervals
        signal_std = df_signals['composite_signal'].std()
        df_signals['signal_upper'] = df_signals['composite_signal'] + 1.96 * signal_std
        df_signals['signal_lower'] = df_signals['composite_signal'] - 1.96 * signal_std
        
        logger.info("Signal generation complete")
        return df_signals

class DataStorage:
    """Handle data storage and retrieval."""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "raw").mkdir(exist_ok=True)
        (self.base_path / "processed").mkdir(exist_ok=True)
        (self.base_path / "signals").mkdir(exist_ok=True)
    
    def save_raw_tweets(self, tweets: List[TweetData], filename: str = None) -> str:
        """Save raw tweets to parquet format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_tweets_{timestamp}.parquet"
        
        filepath = self.base_path / "raw" / filename
        
        # Convert to DataFrame
        tweet_dicts = [tweet.to_dict() for tweet in tweets]
        df = pd.DataFrame(tweet_dicts)
        
        # Save as parquet
        df.to_parquet(filepath, compression='snappy', index=False)
        
        logger.info(f"Saved {len(tweets)} raw tweets to {filepath}")
        return str(filepath)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save processed data to parquet format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.parquet"
        
        filepath = self.base_path / "processed" / filename
        
        # Optimize data types for storage
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except (ValueError, TypeError):
                    pass
        
        # Save with compression
        df.to_parquet(filepath, compression='snappy', index=False)
        
        logger.info(f"Saved processed data to {filepath}")
        return str(filepath)
    
    def load_latest_data(self, data_type: str = "processed") -> Optional[pd.DataFrame]:
        """Load the most recent data file."""
        data_dir = self.base_path / data_type
        
        if not data_dir.exists():
            return None
        
        parquet_files = list(data_dir.glob("*.parquet"))
        
        if not parquet_files:
            return None
        
        # Get the most recent file
        latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)
        
        try:
            df = pd.read_parquet(latest_file)
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {latest_file}: {e}")
            return None

class VisualizationEngine:
    """Create memory-efficient visualizations."""
    
    def __init__(self, max_points: int = 10):
        self.max_points = max_points
        plt.style.use('seaborn-v0_8')
    
    def sample_data_efficiently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample data efficiently for visualization."""
        if len(df) <= self.max_points:
            return df
        
        # Stratified sampling to maintain distribution
        sample_size = min(self.max_points, len(df))
        return df.sample(n=sample_size, random_state=42)
    
    def plot_sentiment_timeline(self, df: pd.DataFrame, save_path: str = None):
        """Plot sentiment timeline with streaming capability."""
        sampled_df = self.sample_data_efficiently(df)
        
        # Group by hour for timeline
        hourly_sentiment = sampled_df.groupby(
            sampled_df['timestamp'].dt.floor('H')
        ).agg({
            'sentiment_signal': ['mean', 'std', 'count'],
            'composite_signal': ['mean', 'std']
        }).reset_index()
        
        hourly_sentiment.columns = [
            'timestamp', 'sentiment_mean', 'sentiment_std', 'tweet_count',
            'composite_mean', 'composite_std'
        ]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Sentiment timeline
        ax1.plot(hourly_sentiment['timestamp'], hourly_sentiment['sentiment_mean'], 
                color='blue', linewidth=2, label='Sentiment Signal')
        ax1.fill_between(
            hourly_sentiment['timestamp'],
            hourly_sentiment['sentiment_mean'] - hourly_sentiment['sentiment_std'],
            hourly_sentiment['sentiment_mean'] + hourly_sentiment['sentiment_std'],
            alpha=0.3, color='blue'
        )
        ax1.set_title('Market Sentiment Timeline (Last 24 Hours)')
        ax1.set_ylabel('Sentiment Signal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tweet volume
        ax2.bar(hourly_sentiment['timestamp'], hourly_sentiment['tweet_count'],
                color='green', alpha=0.7, label='Tweet Volume')
        ax2.set_title('Tweet Volume by Hour')
        ax2.set_ylabel('Number of Tweets')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Composite signal
        ax3.plot(hourly_sentiment['timestamp'], hourly_sentiment['composite_mean'],
                color='red', linewidth=2, label='Composite Signal')
        ax3.fill_between(
            hourly_sentiment['timestamp'],
            hourly_sentiment['composite_mean'] - hourly_sentiment['composite_std'],
            hourly_sentiment['composite_mean'] + hourly_sentiment['composite_std'],
            alpha=0.3, color='red'
        )
        ax3.set_title('Composite Trading Signal')
        ax3.set_ylabel('Signal Strength')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sentiment timeline saved to {save_path}")
        
        return fig
    
    def plot_engagement_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Plot engagement analysis."""
        sampled_df = self.sample_data_efficiently(df)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Engagement vs Followers
        ax1.scatter(np.log1p(sampled_df['followers_count']), 
                   np.log1p(sampled_df['total_engagement']),
                   alpha=0.6, c=sampled_df['sentiment_signal'], cmap='RdYlBu')
        ax1.set_xlabel('Log(Followers + 1)')
        ax1.set_ylabel('Log(Total Engagement + 1)')
        ax1.set_title('Engagement vs Followers (colored by sentiment)')
        
        # Hashtag distribution
        all_hashtags = []
        for hashtags in sampled_df['hashtags']:
            all_hashtags.extend(hashtags)
        
        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
        ax2.barh(range(len(hashtag_counts)), hashtag_counts.values)
        ax2.set_yticks(range(len(hashtag_counts)))
        ax2.set_yticklabels(hashtag_counts.index)
        ax2.set_title('Top 10 Hashtags')
        ax2.set_xlabel('Frequency')
        
        # Hourly activity
        hourly_activity = sampled_df.groupby(sampled_df['timestamp'].dt.hour).size()
        ax3.plot(hourly_activity.index, hourly_activity.values, marker='o')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Tweets')
        ax3.set_title('Tweet Activity by Hour')
        ax3.grid(True, alpha=0.3)
        
        # Signal distribution
        ax4.hist(sampled_df['composite_signal'], bins=30, alpha=0.7, color='purple')
        ax4.axvline(sampled_df['composite_signal'].mean(), color='red', 
                   linestyle='--', label='Mean')
        ax4.set_xlabel('Composite Signal')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Composite Signals')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Engagement analysis saved to {save_path}")
        
        return fig

class MarketIntelligenceSystem:
    """Main system orchestrator."""
    
    def __init__(self):
        self.collector = TwitterCollector()
        self.processor = DataProcessor()
        self.signal_generator = SignalGenerator()
        self.storage = DataStorage()
        self.visualizer = VisualizationEngine()
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("Could not download NLTK data")
    
    def run_collection_cycle(self, target_tweets: int = 2000) -> str:
        """Run a complete data collection and analysis cycle."""
        logger.info("Starting market intelligence collection cycle")
        
        try:
            # 1. Collect tweets
            logger.info("Phase 1: Collecting tweets")
            tweets = self.collector.collect_market_tweets(target_tweets)
            
            if not tweets:
                logger.error("No tweets collected")
                return None
            
            # 2. Save raw data
            raw_file = self.storage.save_raw_tweets(tweets)
            
            # 3. Process tweets
            logger.info("Phase 2: Processing tweets")
            processed_df = self.processor.process_tweets(tweets)
            
            # 4. Generate signals
            logger.info("Phase 3: Generating trading signals")
            signals_df = self.signal_generator.generate_composite_signals(processed_df)
            
            # 5. Save processed data
            processed_file = self.storage.save_processed_data(signals_df)
            
            # 6. Generate visualizations
            logger.info("Phase 4: Creating visualizations")
            
            # Create output directory for plots
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            sentiment_plot = self.visualizer.plot_sentiment_timeline(
                signals_df, 
                save_path=plots_dir / f"sentiment_timeline_{timestamp}.png"
            )
            
            engagement_plot = self.visualizer.plot_engagement_analysis(
                signals_df,
                save_path=plots_dir / f"engagement_analysis_{timestamp}.png"
            )
            
            # 7. Generate summary report
            report = self.generate_analysis_report(signals_df)
            
            # Save report
            report_file = plots_dir / f"analysis_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Collection cycle complete. Processed file: {processed_file}")
            
            return processed_file
            
        except Exception as e:
            logger.error(f"Error in collection cycle: {e}")
            raise
    
    def generate_analysis_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("="*50)
        report.append("MARKET INTELLIGENCE ANALYSIS REPORT")
        report.append("="*50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: Last 24 hours")
        report.append("")
        
        # Collection Statistics
        report.append("COLLECTION STATISTICS")
        report.append("-" * 20)
        report.append(f"Total Tweets Collected: {len(df):,}")
        report.append(f"Unique Users: {df['username'].nunique():,}")
        report.append(f"API Calls Made: {self.collector.collection_stats['api_calls_made']:,}")
        report.append(f"Duplicates Filtered: {self.collector.collection_stats['duplicates_filtered']:,}")
        report.append(f"Rate Limits Hit: {self.collector.collection_stats['rate_limits_hit']:,}")
        report.append("")
        
        # Market Sentiment Summary
        report.append("MARKET SENTIMENT SUMMARY")
        report.append("-" * 25)
        avg_sentiment = df['sentiment_signal'].mean()
        sentiment_direction = "BULLISH" if avg_sentiment > 0 else "BEARISH" if avg_sentiment < 0 else "NEUTRAL"
        
        report.append(f"Overall Sentiment: {sentiment_direction}")
        report.append(f"Average Sentiment Score: {avg_sentiment:.4f}")
        report.append(f"Sentiment Standard Deviation: {df['sentiment_signal'].std():.4f}")
        report.append(f"Bullish Tweets: {(df['sentiment_signal'] > 0).sum():,} ({(df['sentiment_signal'] > 0).mean()*100:.1f}%)")
        report.append(f"Bearish Tweets: {(df['sentiment_signal'] < 0).sum():,} ({(df['sentiment_signal'] < 0).mean()*100:.1f}%)")
        report.append("")
        
        # Engagement Metrics
        report.append("ENGAGEMENT METRICS")
        report.append("-" * 18)
        report.append(f"Total Engagement: {df['total_engagement'].sum():,}")
        report.append(f"Average Engagement per Tweet: {df['total_engagement'].mean():.2f}")
        report.append(f"Highest Engagement Tweet: {df['total_engagement'].max():,}")
        report.append(f"Most Followed User: {df['followers_count'].max():,} followers")
        report.append("")
        
        # Top Hashtags
        report.append("TOP HASHTAGS")
        report.append("-" * 12)
        all_hashtags = []
        for hashtags in df['hashtags']:
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        
        if all_hashtags:
            top_hashtags = pd.Series(all_hashtags).value_counts().head(10)
            for i, (hashtag, count) in enumerate(top_hashtags.items(), 1):
                report.append(f"{i:2d}. #{hashtag}: {count:,} mentions")
        report.append("")
        
        # Trading Signals
        report.append("TRADING SIGNALS")
        report.append("-" * 15)
        current_signal = df['composite_signal'].iloc[-100:].mean()  # Last 100 tweets
        signal_trend = "INCREASING" if df['composite_signal'].diff().iloc[-50:].mean() > 0 else "DECREASING"
        
        report.append(f"Current Composite Signal: {current_signal:.4f}")
        report.append(f"Signal Trend: {signal_trend}")
        report.append(f"Signal Volatility: {df['composite_signal'].std():.4f}")
        
        # Signal strength interpretation
        if abs(current_signal) > df['composite_signal'].std() * 2:
            strength = "STRONG"
        elif abs(current_signal) > df['composite_signal'].std():
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        direction = "BULLISH" if current_signal > 0 else "BEARISH"
        report.append(f"Signal Strength: {strength} {direction}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        if abs(current_signal) > df['composite_signal'].std() * 1.5:
            report.append(f"• Strong {direction.lower()} signal detected")
            report.append("• Consider position adjustment based on signal direction")
        else:
            report.append("• Neutral to weak signals - maintain current positions")
        
        report.append("• Monitor for signal strength changes in next collection cycle")
        report.append("• High tweet volume may indicate increased market volatility")
        report.append("")
        
        # Data Quality Metrics
        report.append("DATA QUALITY METRICS")
        report.append("-" * 20)
        report.append(f"Data Completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))*100:.2f}%")
        report.append(f"Average Content Length: {df['length'].mean():.1f} characters")
        report.append(f"Language Distribution: {df['language'].value_counts().to_dict()}")
        report.append("")
        
        return "\n".join(report)
    
    def run_continuous_monitoring(self, interval_minutes: int = 30):
        """Run continuous monitoring and data collection."""
        logger.info(f"Starting continuous monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                logger.info("Running scheduled collection cycle")
                self.run_collection_cycle()
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Continuous monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

class StreamingVisualizer:
    """Real-time streaming visualizations."""
    
    def __init__(self, update_interval: int = 5000):  # 5 seconds
        self.update_interval = update_interval
        self.data_queue = queue.Queue(maxsize=10)
        self.fig = None
        self.ax = None
        
    def start_streaming_plot(self, df: pd.DataFrame):
        """Start streaming sentiment plot."""
        plt.ion()  # Turn on interactive mode
        
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Initialize with existing data
        timestamps = df['timestamp'].values[-100:]  # Last 100 tweets
        signals = df['composite_signal'].values[-100:]
        
        line, = self.ax.plot(timestamps, signals, 'b-', linewidth=2)
        self.ax.set_title('Real-time Market Sentiment Signal')
        self.ax.set_ylabel('Signal Strength')
        self.ax.grid(True, alpha=0.3)
        
        def animate(frame):
            try:
                # Get new data from queue
                while not self.data_queue.empty():
                    new_data = self.data_queue.get_nowait()
                    timestamps = np.append(timestamps, new_data['timestamp'])
                    signals = np.append(signals, new_data['signal'])
                    
                    # Keep only last 100 points for memory efficiency
                    if len(timestamps) > 100:
                        timestamps = timestamps[-100:]
                        signals = signals[-100:]
                
                # Update plot
                line.set_data(timestamps, signals)
                self.ax.relim()
                self.ax.autoscale_view()
                
            except queue.Empty:
                pass
            
            return line,
        
        ani = FuncAnimation(self.fig, animate, interval=self.update_interval, blit=True)
        plt.show()
        
        return ani
    
    def add_data_point(self, timestamp: datetime, signal: float):
        """Add new data point to streaming plot."""
        try:
            self.data_queue.put_nowait({
                'timestamp': timestamp,
                'signal': signal
            })
        except queue.Full:
            logger.warning("Streaming data queue is full")

# Utility functions
def load_environment_variables():
    """Load environment variables from multiple sources."""
    # Try to load from .env file first
    env_file = Path('.env')
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")
        except Exception as e:
            logger.warning(f"Error loading .env file: {e}")
    
    # Check if environment variables are set
    required_env_vars = [
        'TWITTER_API_KEY',
        'TWITTER_API_SECRET', 
        'TWITTER_ACCESS_TOKEN',
        'TWITTER_ACCESS_TOKEN_SECRET',
        'TWITTER_BEARER_TOKEN'
    ]
    
    env_status = {}
    for var in required_env_vars:
        value = os.getenv(var)
        env_status[var] = "✓ Set" if value else "✗ Missing"
        if value:
            logger.info(f"{var}: {value[:10]}..." if len(value) > 10 else f"{var}: {value}")
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Environment Variables Status:")
        for var, status in env_status.items():
            logger.error(f"  {var}: {status}")
        logger.error("")
        logger.error("Solutions:")
        logger.error("1. Create a .env file in the current directory with:")
        logger.error("   TWITTER_API_KEY=your_key_here")
        logger.error("   TWITTER_API_SECRET=your_secret_here") 
        logger.error("   TWITTER_ACCESS_TOKEN=your_token_here")
        logger.error("   TWITTER_ACCESS_TOKEN_SECRET=your_token_secret_here")
        logger.error("   TWITTER_BEARER_TOKEN=your_bearer_token_here")
        logger.error("")
        logger.error("2. Or set them in your system environment")
        logger.error("3. Or pass them as command line arguments")
        
        raise ValueError(f"Missing environment variables: {missing_vars}")
    
    logger.info("All required environment variables are set")
    return True

def setup_environment():
    """Setup the environment and check dependencies."""
    # Load environment variables
    load_environment_variables()
    
    # Create necessary directories
    directories = ['data', 'data/raw', 'data/processed', 'data/signals', 'plots', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("Environment setup complete")

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return metrics."""
    quality_metrics = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_tweets': df['tweet_id'].duplicated().sum(),
        'data_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'time_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        },
        'language_distribution': df['language'].value_counts().to_dict(),
        'engagement_stats': {
            'avg_engagement': df['total_engagement'].mean(),
            'max_engagement': df['total_engagement'].max(),
            'std_engagement': df['total_engagement'].std()
        }
    }
    
    return quality_metrics

def export_signals_for_trading(df: pd.DataFrame, output_file: str = "trading_signals.json"):
    """Export signals in format suitable for trading systems."""
    
    # Calculate aggregated signals
    latest_hour = df['timestamp'].max().floor('H')
    recent_tweets = df[df['timestamp'] >= latest_hour - pd.Timedelta(hours=1)]
    
    if len(recent_tweets) == 0:
        logger.warning("No recent tweets found for signal export")
        return
    
    signals = {
        'timestamp': datetime.now().isoformat(),
        'data_points': len(recent_tweets),
        'sentiment_signal': {
            'value': float(recent_tweets['sentiment_signal'].mean()),
            'confidence': float(1 / (1 + recent_tweets['sentiment_signal'].std())),
            'direction': 'bullish' if recent_tweets['sentiment_signal'].mean() > 0 else 'bearish'
        },
        'volume_signal': {
            'value': float(len(recent_tweets)),
            'percentile': float(len(recent_tweets) / len(df) * 100)
        },
        'composite_signal': {
            'value': float(recent_tweets['composite_signal'].mean()),
            'upper_bound': float(recent_tweets['signal_upper'].mean()),
            'lower_bound': float(recent_tweets['signal_lower'].mean()),
            'volatility': float(recent_tweets['composite_signal'].std())
        },
        'top_mentions': recent_tweets['mentions'].explode().value_counts().head(5).to_dict(),
        'trending_hashtags': recent_tweets['hashtags'].explode().value_counts().head(10).to_dict()
    }
    
    # Save signals
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)
    
    logger.info(f"Trading signals exported to {output_file}")
    return signals

# CLI Interface
def main():
    """Main function to run the market intelligence system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Intelligence System")
    parser.add_argument('--collect', action='store_true', help='Run data collection')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing data')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--target-tweets', type=int, default=100, help='Target number of tweets')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in minutes')
    parser.add_argument('--export-signals', action='store_true', help='Export trading signals')
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize system
        system = MarketIntelligenceSystem()
        
        if args.collect or (not args.analyze and not args.continuous):
            # Run collection cycle
            logger.info(f"Starting data collection (target: {args.target_tweets} tweets)")
            processed_file = system.run_collection_cycle(args.target_tweets)
            
            if processed_file and args.export_signals:
                # Load and export signals
                df = system.storage.load_latest_data("processed")
                if df is not None:
                    export_signals_for_trading(df)
        
        elif args.analyze:
            # Analyze existing data
            logger.info("Analyzing existing data")
            df = system.storage.load_latest_data("processed")
            
            if df is None:
                logger.error("No processed data found. Run collection first.")
                return
            
            # Generate new visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            
            system.visualizer.plot_sentiment_timeline(
                df, 
                save_path=plots_dir / f"sentiment_analysis_{timestamp}.png"
            )
            
            system.visualizer.plot_engagement_analysis(
                df,
                save_path=plots_dir / f"engagement_analysis_{timestamp}.png"
            )
            
            # Generate report
            report = system.generate_analysis_report(df)
            print(report)
            
            if args.export_signals:
                export_signals_for_trading(df)
        
        elif args.continuous:
            # Run continuous monitoring
            logger.info(f"Starting continuous monitoring (every {args.interval} minutes)")
            system.run_continuous_monitoring(args.interval)
    
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    main()

# Additional utility classes for advanced features

class AdvancedTextProcessor:
    """Advanced text processing for better signal generation."""
    
    def __init__(self):
        self.market_lexicon = self._load_market_lexicon()
        self.emoji_sentiment = self._load_emoji_sentiment()
    
    def _load_market_lexicon(self) -> Dict[str, float]:
        """Load market-specific sentiment lexicon."""
        # Market-specific sentiment words with scores
        return {
            'moon': 2.0, 'rocket': 2.0, 'bullish': 1.5, 'buy': 1.0,
            'target': 1.0, 'breakout': 1.5, 'rally': 1.5, 'support': 0.5,
            'crash': -2.0, 'dump': -2.0, 'bearish': -1.5, 'sell': -1.0,
            'fall': -1.0, 'resistance': -0.5, 'correction': -1.0,
            'hodl': 1.0, 'diamond': 1.5, 'hands': 1.0, 'paper': -1.0
        }
    
    def _load_emoji_sentiment(self) -> Dict[str, float]:
        """Load emoji sentiment mapping."""
        return {
            '🚀': 2.0, '📈': 1.5, '💎': 1.5, '🌙': 2.0, '💪': 1.0,
            '📉': -1.5, '😭': -1.0, '💸': -1.5, '🔥': 1.0, '⚡': 1.0
        }
    
    def calculate_advanced_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate advanced sentiment scores."""
        text_lower = text.lower()
        
        # Market lexicon sentiment
        lexicon_score = sum(
            score for word, score in self.market_lexicon.items()
            if word in text_lower
        )
        
        # Emoji sentiment
        emoji_score = sum(
            score for emoji, score in self.emoji_sentiment.items()
            if emoji in text
        )
        
        # Caps lock intensity (often indicates strong emotion)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        caps_score = caps_ratio * 0.5 if caps_ratio > 0.3 else 0
        
        return {
            'lexicon_sentiment': lexicon_score,
            'emoji_sentiment': emoji_score,
            'caps_intensity': caps_score,
            'combined_sentiment': lexicon_score + emoji_score + caps_score
        }

class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.metrics = {
            'collection_times': [],
            'processing_times': [],
            'memory_usage': [],
            'api_response_times': []
        }
    
    def start_timer(self) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_timer(self, start_time: float, operation: str):
        """End timing and record the operation."""
        duration = time.time() - start_time
        
        if operation in self.metrics:
            self.metrics[operation].append(duration)
        
        logger.info(f"{operation} completed in {duration:.2f} seconds")
        return duration
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'avg_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'total_operations': len(times)
                }
        
        return summary

class ConfigManager:
    """Manage system configuration."""
    
    DEFAULT_CONFIG = {
        'collection': {
            'target_tweets': 2000,
            'max_tweets_per_hashtag': 300,
            'collection_interval_minutes': 30,
            'max_concurrent_requests': 3
        },
        'processing': {
            'max_text_length': 500,
            'min_engagement_threshold': 0,
            'sentiment_smoothing_window': 10
        },
        'storage': {
            'compression': 'snappy',
            'max_file_size_mb': 100,
            'backup_enabled': True
        },
        'visualization': {
            'max_plot_points': 10,
            'update_interval_ms': 5000,
            'color_scheme': 'viridis'
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        # Save default config
        self.save_config(self.DEFAULT_CONFIG)
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file."""
        config_to_save = config or self.config
        
        with open(self.config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        logger.info(f"Configuration saved to {self.config_file}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
if __name__ == "__main__":
    print("Market Intelligence System")
    print("="*30)
    
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("python market_intelligence.py --collect")
        print("python market_intelligence.py --analyze")
        print("python market_intelligence.py --continuous --interval 30")
        print("python market_intelligence.py --collect --export-signals")
        print()
        print("- TWITTER_API_KEY")
        print("- TWITTER_API_SECRET") 
        print("- TWITTER_ACCESS_TOKEN")
        print("- TWITTER_ACCESS_TOKEN_SECRET")
        print("- TWITTER_BEARER_TOKEN")
    else:
        main()