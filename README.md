# Market Intelligence Backend

## Overview
This repository contains a Python-based backend for a real-time market intelligence system, focusing on collecting and analyzing Indian stock market discussions from Twitter/X. The system leverages **Tweepy** for data collection and is designed for scalability and production use.

## Features
- Scrapes Twitter/X for tweets using hashtags like `#nifty50`, `#sensex`, `#intraday`, and `#banknifty`.
- Processes and stores data efficiently in Parquet format.
- Generates trading signals from tweet content using text analysis techniques.
- Includes logging and configuration management for production readiness.

## Repository Structure

backend/
├── __pycache__/          # Python bytecode cache
├── .env/                 # Twitter api keys
├── config.json           # Configuration settings 
├── main.py               # Main script to run the backend
├── market_intelligence.log # Log file for system activity
├── README.md             # Project overview and setup
└── requirements.txt      # Python dependencies


## Prerequisites
- Python 3.8+
- Twitter/X API credentials
- A modern browser for potential visualization outputs (optional)

## Setup Instructions
1. **Clone the Repository**:
   bash
   git clone https://github.com/Alok214/MarketIntelligence.git
   cd MarketIntelligence
   

2. **Set Up Virtual Environment**:
   bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   

3. **Install Dependencies**:
   bash
   pip install -r requirements.txt
   

4. **Configure API Credentials**:
   Edit `config.json` with your Twitter/X API credentials:
   json
   {
     "twitter_api_key": "your_api_key",
     "twitter_api_secret": "your_api_secret",
     "twitter_access_token": "your_access_token",
     "twitter_access_token_secret": "your_access_token_secret"
   }
   

5. **Run the Backend**:
   bash
   python main.py
   

6. **Check Logs**:
   - System logs are saved to `market_intelligence.log` for debugging and monitoring.

## Dependencies
See `requirements.txt` for a full list. Key dependencies include:
- `tweepy`: For Twitter/X API interaction
- `pandas`: For data processing
- `logging`: For log management

## Usage
- Run `main.py` to start the backend, which handles tweet collection, processing, and logging.
- The script is designed to manage rate limits and store intermediate data (further storage details may be added in future iterations).

## Notes
- The system avoids paid APIs by using Tweepy's free tier.
- Logs are written to `market_intelligence.log` for tracking performance and errors.
