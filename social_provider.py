import abc
import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

AUTH_STATE_FILE = ".auth_state.json"

class SocialDataProvider(abc.ABC):
    """
    Abstract Base Class for social media sentiment data ingestion.
    """
    @abc.abstractmethod
    def authenticate(self):
        """Prompt user to authenticate if required."""
        pass

    @abc.abstractmethod
    def get_social_chatter(self, ticker: str, lookback_days: int = 7) -> List[Dict[str, str]]:
        """
        Fetch social media chatter for a given ticker.
        Returns a list of dictionaries with keys: 'text', 'source', 'timestamp'
        """
        pass

class BrowserScraperProvider(SocialDataProvider):
    """
    Uses Playwright to scrape Reddit and X search pages via an authenticated browser session.
    """
    def __init__(self, auth_file: str = AUTH_STATE_FILE):
        self.auth_file = auth_file

    def authenticate(self):
        """
        Launches a non-headless browser to allow the user to log into X and Reddit manually.
        Saves the resulting state to a local JSON file.
        """
        print(f"\n[BrowserScraper] Launching browser for authentication...")
        print("Please log into Reddit and X. Close the browser when finished.")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            
            # If we already have a state, load it to avoid re-logging in if possible
            if os.path.exists(self.auth_file):
                context = browser.new_context(storage_state=self.auth_file)
            else:
                context = browser.new_context()

            page = context.new_page()
            
            # Open Reddit and X in separate tabs to prompt user
            page.goto("https://www.reddit.com/login/")
            page2 = context.new_page()
            page2.goto("https://twitter.com/i/flow/login")
            
            # Keep browser open until user closes it
            try:
                page.wait_for_event("close", timeout=0)
            except Exception:
                pass # Event will fire when user closes

            # Save state
            context.storage_state(path=self.auth_file)
            browser.close()
            print(f"[BrowserScraper] Authentication state saved to {self.auth_file}.\n")

    def _scrape_reddit(self, page, ticker: str, lookback_days: int) -> List[Dict[str, str]]:
        results = []
        try:
            # Sort by new
            url = f"https://www.reddit.com/search/?q={ticker}&sort=new"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000) # Give time for React to render

            # Simple heuristic: grab post titles and snippets
            # Note: Reddit DOM changes often, so we use broad selectors
            posts = page.locator('a[data-testid="post-title"]')
            snippets = page.locator('div[data-testid="post-content"]')
            
            count = posts.count()
            for i in range(min(count, 15)): # Limit to top 15 posts
                title = posts.nth(i).text_content()
                if title:
                    results.append({
                        'source': 'Reddit',
                        'text': title.strip(),
                        'timestamp': datetime.now().isoformat() # Approximated, as parsing Reddit relative time is complex
                    })
        except Exception as e:
            print(f"Failed to scrape Reddit for {ticker}: {e}")
        return results

    def _scrape_x(self, page, ticker: str, lookback_days: int) -> List[Dict[str, str]]:
        results = []
        try:
            url = f"https://twitter.com/search?q={ticker}&f=live"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

            tweets = page.locator('div[data-testid="tweetText"]')
            count = tweets.count()
            for i in range(min(count, 15)):
                text = tweets.nth(i).text_content()
                if text:
                    results.append({
                        'source': 'X',
                        'text': text.strip(),
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Failed to scrape X for {ticker}: {e}")
        return results

    def get_social_chatter(self, ticker: str, lookback_days: int = 7) -> List[Dict[str, str]]:
        """Scrapes Reddit and X in headless mode using saved auth state."""
        chatter = []
        
        if not os.path.exists(self.auth_file):
            print(f"[Warning] {self.auth_file} not found. Running unauthenticated scrape, results may be limited.")
            # Depending on use case, could force authenticate() here, but we will try unauthenticated
            
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            
            if os.path.exists(self.auth_file):
                context = browser.new_context(storage_state=self.auth_file)
            else:
                context = browser.new_context()

            page = context.new_page()
            
            print(f"Scraping social chatter for {ticker} (lookback: {lookback_days} days)...")
            reddit_data = self._scrape_reddit(page, ticker, lookback_days)
            x_data = self._scrape_x(page, ticker, lookback_days)
            
            chatter.extend(reddit_data)
            chatter.extend(x_data)
            
            browser.close()
            
        return chatter
