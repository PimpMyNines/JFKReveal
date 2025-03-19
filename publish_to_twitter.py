#!/usr/bin/env python
"""
Script to publish JFK analysis report to Twitter/X as a thread
Uses Twitter API v2 with a Personal Access Token (PAT)
"""

import os
import argparse
import time
import logging
from typing import List, Dict, Optional, Tuple
import json
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterPublisher:
    """Publishes content to Twitter/X using the Twitter API v2."""
    
    def __init__(self, bearer_token: str):
        """
        Initialize the Twitter publisher.
        
        Args:
            bearer_token: Personal Access Token (PAT) for Twitter API
        """
        self.bearer_token = bearer_token
        self.api_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
    
    def create_tweet(self, text: str, media_ids: Optional[List[str]] = None, 
                     reply_to: Optional[str] = None) -> Dict:
        """
        Create a new tweet.
        
        Args:
            text: Tweet text content
            media_ids: List of uploaded media IDs to attach
            reply_to: Tweet ID to reply to (for creating threads)
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.api_url}/tweets"
        
        payload = {"text": text}
        
        if media_ids:
            payload["media"] = {"media_ids": media_ids}
            
        if reply_to:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}
            
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating tweet: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def upload_media(self, media_path: str) -> str:
        """
        Upload media (image) to Twitter.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            Media ID
        """
        # Twitter API v2 media upload endpoint
        url = "https://upload.twitter.com/1.1/media/upload.json"
        
        # Read the image file
        with open(media_path, "rb") as file:
            files = {"media": file}
            
            try:
                response = requests.post(
                    url, 
                    headers={"Authorization": f"Bearer {self.bearer_token}"},
                    files=files
                )
                response.raise_for_status()
                return response.json()["media_id_string"]
            except requests.exceptions.RequestException as e:
                logger.error(f"Error uploading media: {e}")
                if hasattr(e, 'response') and e.response:
                    logger.error(f"Response: {e.response.text}")
                raise

def create_thread_content() -> List[Dict]:
    """
    Create content for the Twitter thread.
    
    Returns:
        List of dictionaries with tweet text and optional image path
    """
    thread = [
        {
            "text": "I used @ClaudeAI to analyze declassified JFK assassination documents. Here's what the AI found: [THREAD]\n\nThanks to the National Archives, NARA, and JFK Records Act for making these documents public.\n\n#JFKFiles #AI #History",
            "image": "images/jfk_portrait.jpg"  # Replace with actual image path
        },
        {
            "text": "Executive Summary: The documents reveal Lee Harvey Oswald as primary suspect but suggest possible involvement from the Mafia, Soviet Union, CIA, and Cuban intelligence. Significant redactions complicate definitive conclusions.\n\n#JFKAssassination #History"
        },
        {
            "text": "Key Evidence:\n• Oswald's presence at Texas School Book Depository\n• His connections to pro-Castro groups and Soviet Union\n• Mafia figures discussing assassination\n• CIA surveillance of Oswald in Mexico City\n• Witness testimonies suggesting multiple shooters",
            "image": "images/book_depository.jpg"  # Replace with actual image path
        },
        {
            "text": "Potential Government Involvement:\n• Redacted sections about CIA/FBI surveillance\n• Allegations of pressure on Warren Commission\n• Evidence suppression regarding bullet trajectory\n• Altered/destroyed Secret Service reports",
            "image": "images/redacted_document.jpg"  # Replace with actual image path
        },
        {
            "text": "Most Credible Theories:\n• Oswald as lone gunman\n• Mafia involvement (JFK's crackdown on organized crime)\n• Multiple shooters theory\n• CIA's potential foreknowledge of Oswald's activities",
            "image": "images/conspiracy_map.jpg"  # Replace with actual image path
        },
        {
            "text": "The full analysis and code is now open source at [YOUR GITHUB URL]\n\nThis project uses @LangChainAI and OpenAI to analyze documents - feel free to expand on this work!\n\n🤖 Analysis by @ClaudeAI Code\n📄 Data from National Archives #JFKFiles",
            "image": "images/report_screenshot.jpg"  # Replace with actual image path
        }
    ]
    
    return thread

def extract_tweet_content_from_report(report_path: str, max_length: int = 275) -> List[Dict]:
    """
    Extract tweet content from a report markdown file.
    
    Args:
        report_path: Path to the markdown report
        max_length: Maximum tweet length
        
    Returns:
        List of dictionaries with tweet text
    """
    with open(report_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by headers
    sections = []
    current_section = ""
    
    for line in content.split('\n'):
        if line.startswith('# ') or line.startswith('## '):
            if current_section:
                sections.append(current_section.strip())
            current_section = line
        else:
            current_section += "\n" + line
    
    if current_section:
        sections.append(current_section.strip())
    
    # Convert sections to tweets
    tweets = []
    
    # First tweet is introduction
    tweets.append({
        "text": "I used @ClaudeAI to analyze declassified JFK assassination documents. Here's what the AI found: [THREAD]\n\nThanks to the National Archives, NARA, and JFK Records Act for making these documents public.\n\n#JFKFiles #AI #History",
        "image": "images/jfk_portrait.jpg"  # Replace with actual image path
    })
    
    for section in sections:
        # Get the section title
        lines = section.split('\n')
        title = lines[0].replace('#', '').strip()
        
        # Process the section content
        content = '\n'.join(lines[1:]).strip()
        
        # If it's a bullet list, extract and truncate
        if '- ' in content:
            bullet_points = [line.strip() for line in content.split('\n') if line.strip().startswith('- ')]
            bullet_text = '\n'.join(bullet_points[:5])  # Limit to 5 bullet points
            
            tweet_text = f"{title}:\n{bullet_text}"
            if len(tweet_text) > max_length:
                tweet_text = tweet_text[:max_length-3] + "..."
        else:
            # For regular text, truncate as needed
            content_short = content[:max_length-len(title)-3]
            if content != content_short:
                content_short += "..."
                
            tweet_text = f"{title}:\n{content_short}"
        
        tweets.append({"text": tweet_text})
    
    # Last tweet is about the repo
    tweets.append({
        "text": "The full analysis and code is now open source at [YOUR GITHUB URL]\n\nThis project uses @LangChainAI and OpenAI to analyze documents - feel free to expand on this work!\n\n🤖 Analysis by @ClaudeAI Code\n📄 Data from National Archives #JFKFiles",
        "image": "images/report_screenshot.jpg"  # Replace with actual image path
    })
    
    return tweets

def create_images_directory():
    """Create directory for tweet images if it doesn't exist."""
    os.makedirs("images", exist_ok=True)
    logger.info("Created images directory. Please add your images there before posting.")
    
    # Create a reminder file
    with open("images/README.txt", "w") as f:
        f.write("""Please add the following images to this directory:
1. jfk_portrait.jpg - A portrait of JFK or motorcade photo
2. book_depository.jpg - Texas School Book Depository image
3. redacted_document.jpg - Image of redacted JFK documents
4. conspiracy_map.jpg - Visual representation of theories
5. report_screenshot.jpg - Screenshot of your report

These will be attached to your tweets when posting.""")

def main():
    """Main function to publish the thread to Twitter."""
    parser = argparse.ArgumentParser(description="Publish JFK analysis report to Twitter/X")
    parser.add_argument("--token", help="Twitter API Bearer Token (PAT)")
    parser.add_argument("--report", help="Path to executive summary markdown file", 
                        default="data/reports/executive_summary.md")
    parser.add_argument("--use-default", action="store_true", 
                        help="Use default thread content instead of extracting from report")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print tweets without publishing")
    args = parser.parse_args()
    
    # Check for token
    bearer_token = args.token or os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token and not args.dry_run:
        logger.error("No Twitter Bearer Token provided. Use --token or set TWITTER_BEARER_TOKEN environment variable.")
        return
    
    # Create images directory if it doesn't exist
    create_images_directory()
    
    # Get thread content
    if args.use_default:
        thread_content = create_thread_content()
        logger.info("Using default thread content")
    else:
        report_path = args.report
        if not os.path.exists(report_path):
            logger.error(f"Report file not found: {report_path}")
            return
        
        thread_content = extract_tweet_content_from_report(report_path)
        logger.info(f"Extracted {len(thread_content)} tweets from report")
    
    # Print tweets in dry run mode
    if args.dry_run:
        logger.info("DRY RUN - Tweets will not be published")
        for i, tweet in enumerate(thread_content, 1):
            logger.info(f"Tweet {i}:")
            logger.info(f"Text: {tweet['text']}")
            if 'image' in tweet:
                logger.info(f"Image: {tweet['image']}")
            logger.info("-" * 40)
        return
    
    # Initialize Twitter publisher
    twitter = TwitterPublisher(bearer_token)
    
    # Publish thread
    previous_tweet_id = None
    
    for i, tweet in enumerate(thread_content, 1):
        logger.info(f"Publishing tweet {i}/{len(thread_content)}...")
        
        # Check for image
        media_ids = []
        if 'image' in tweet and os.path.exists(tweet['image']):
            logger.info(f"Uploading image: {tweet['image']}")
            media_id = twitter.upload_media(tweet['image'])
            media_ids.append(media_id)
            logger.info(f"Media uploaded with ID: {media_id}")
        
        # Create tweet
        response = twitter.create_tweet(
            text=tweet['text'],
            media_ids=media_ids if media_ids else None,
            reply_to=previous_tweet_id
        )
        
        tweet_id = response['data']['id']
        logger.info(f"Tweet published with ID: {tweet_id}")
        previous_tweet_id = tweet_id
        
        # Add delay to avoid rate limits
        if i < len(thread_content):
            logger.info("Waiting 5 seconds before next tweet...")
            time.sleep(5)
    
    logger.info(f"Thread published successfully with {len(thread_content)} tweets!")

if __name__ == "__main__":
    main()