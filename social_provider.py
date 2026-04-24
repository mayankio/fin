import abc
import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
try:
    from playwright_stealth import Stealth
except ImportError:
    Stealth = None

AUTH_STATE_FILE = ".auth_state.json"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

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
            browser = p.firefox.launch(headless=False)
            
            if os.path.exists(self.auth_file):
                context = browser.new_context(storage_state=self.auth_file, user_agent=USER_AGENT, viewport={'width': 1280, 'height': 800})
            else:
                context = browser.new_context(user_agent=USER_AGENT, viewport={'width': 1280, 'height': 800})

            page = context.new_page()
            if Stealth:
                Stealth().apply_stealth_sync(page)
            
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
            # Reddit uses custom <shreddit-post> web components
            try:
                page.wait_for_selector('shreddit-post', timeout=8000)
            except Exception:
                pass # Timeout, maybe no posts or blocked
                
            posts = page.locator('shreddit-post')
            
            count = posts.count()
            for i in range(min(count, 15)): # Limit to top 15 posts
                title = posts.nth(i).get_attribute("post-title")
                if not title:
                    title = posts.nth(i).text_content()
                
                if title:
                    results.append({
                        'source': 'Reddit',
                        'text': title.strip(),
                        'timestamp': datetime.now().isoformat()
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

            try:
                page.wait_for_selector('article', timeout=8000)
            except Exception:
                pass

            tweets = page.locator('article')
            count = tweets.count()
            for i in range(min(count, 15)):
                tweet_text_locator = tweets.nth(i).locator('div[data-testid="tweetText"]')
                if tweet_text_locator.count() > 0:
                    text = tweet_text_locator.nth(0).text_content()
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
            # Launch headful to bypass strict bot blocks
            browser = p.firefox.launch(headless=False)
            
            if os.path.exists(self.auth_file):
                context = browser.new_context(storage_state=self.auth_file, user_agent=USER_AGENT, viewport={'width': 1280, 'height': 800})
            else:
                context = browser.new_context(user_agent=USER_AGENT, viewport={'width': 1280, 'height': 800})

            page = context.new_page()
            if Stealth:
                Stealth().apply_stealth_sync(page)
            
            print(f"Scraping social chatter for {ticker} (lookback: {lookback_days} days)...")
            reddit_data = self._scrape_reddit(page, ticker, lookback_days)
            x_data = self._scrape_x(page, ticker, lookback_days)
            
            chatter.extend(reddit_data)
            chatter.extend(x_data)
            
            browser.close()
            
        return chatter
