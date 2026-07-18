import pytest
from datetime import UTC, datetime

from marketingos.agents.research import (
    ResearchAgent,
    ResearchInput,
    WebsiteSnapshot,
    InstagramProfileSnapshot,
    SearchResultSnapshot,
    ContactDetails,
)


@pytest.fixture
def mock_website_scraper():
    async def scrape(url: str):
        return WebsiteSnapshot(
            url=url,
            title="Test Site",
            about_text="Test business description",
            contact=ContactDetails(emails=("test@example.com",)),
        )

    class MockScraper:
        pass

    mock = MockScraper()
    mock.scrape = scrape
    return mock


@pytest.fixture
def mock_instagram_reader():
    async def fetch_profile(username: str):
        return InstagramProfileSnapshot(
            username=username,
            profile_url=f"https://instagram.com/{username}",
            full_name="Test Account",
            biography="Test bio",
            follower_count=1000,
        )

    class MockReader:
        pass

    mock = MockReader()
    mock.fetch_profile = fetch_profile
    return mock


@pytest.fixture
def mock_search_tool():
    async def search(query: str, *, max_results: int):
        return (
            SearchResultSnapshot(
                title="Test Result",
                url="https://example.com/result",
                snippet="Test snippet about the business",
            ),
        )

    class MockSearch:
        pass

    mock = MockSearch()
    mock.search = search
    return mock


@pytest.mark.asyncio
async def test_research_agent_execute(mock_website_scraper, mock_instagram_reader, mock_search_tool):
    agent = ResearchAgent(
        website_scraper=mock_website_scraper,
        instagram_reader=mock_instagram_reader,
        search_tool=mock_search_tool,
    )

    result = await agent.execute(ResearchInput(website_url="https://example.com"))

    assert result.facts
    assert result.confidence_score is not None
    assert result.run_id
