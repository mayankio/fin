import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from social_provider import BrowserScraperProvider

@pytest.fixture
def provider():
    return BrowserScraperProvider(auth_file="dummy_auth.json")

@patch('social_provider.sync_playwright')
@patch('social_provider.os.path.exists', return_value=True)
def test_get_social_chatter(mock_exists, mock_playwright, provider):
    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_playwright.return_value.__enter__.return_value = mock_p
    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Mock the locators for Reddit and X
    mock_locator = MagicMock()
    mock_locator.count.return_value = 2
    mock_locator.nth.return_value.text_content.side_effect = ["Title 1", "Title 2", "Tweet 1", "Tweet 2"]
    mock_page.locator.return_value = mock_locator

    chatter = provider.get_social_chatter("AAPL", lookback_days=7)

    assert len(chatter) == 4
    assert chatter[0]['source'] == 'Reddit'
    assert chatter[0]['text'] == 'Title 1'
    assert chatter[2]['source'] == 'X'
    assert chatter[2]['text'] == 'Tweet 1'
    
    # Check that it tried to go to both URLs
    assert mock_page.goto.call_count == 2
