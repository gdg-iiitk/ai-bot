import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from typing import List, Union
import logging

class WebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            self.logger.error(f"Invalid URL: {e}")
            return False

    def get_soup(self, endpoint: str = "") -> Union[BeautifulSoup, None]:
        url = urljoin(self.base_url, endpoint)
        if not self.validate_url(url):
            return None
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def scrape_by_tag(self, tag: str, endpoint: str = "", attributes: dict = None) -> List[str]:
        soup = self.get_soup(endpoint)
        if not soup:
            return []
        
        elements = soup.find_all(tag, attributes or {})
        return [elem.text.strip() for elem in elements]

    def scrape_by_class(self, class_name: str, endpoint: str = "") -> List[str]:
        return self.scrape_by_tag(tag="*", endpoint=endpoint, attributes={"class": class_name})

    def scrape_by_id(self, id_value: str, endpoint: str = "") -> str:
        soup = self.get_soup(endpoint)
        if not soup:
            return ""
        
        element = soup.find(id=id_value)
        return element.text.strip() if element else ""

    def save_to_file(self, data: Union[str, List[str]], filename: str):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                if isinstance(data, list):
                    f.write('\n'.join(data))
                else:
                    f.write(data)
            self.logger.info(f"Data saved to {filename}")
        except IOError as e:
            self.logger.error(f"Failed to save data: {e}")

class SiteMapper:
    def __init__(self, scraper: WebScraper):
        self.scraper = scraper
        self.visited = set()

    def get_all_links(self, endpoint: str = "") -> List[str]:
        soup = self.scraper.get_soup(endpoint)
        if not soup:
            return []

        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(self.scraper.base_url, href)
            if full_url.startswith(self.scraper.base_url):
                links.append(full_url)
        return list(set(links))

    def map_site(self, endpoint: str = "", max_depth: int = 3) -> List[str]:
        if max_depth <= 0 or endpoint in self.visited:
            return []

        self.visited.add(endpoint)
        links = self.get_all_links(endpoint)
        
        all_endpoints = [endpoint]
        for link in links:
            relative_path = urlparse(link).path
            all_endpoints.extend(self.map_site(relative_path, max_depth - 1))
        
        return list(set(all_endpoints))

# Example usage
if __name__ == "__main__":
    scraper = WebScraper("https://example.com")
    mapper = SiteMapper(scraper)
    
    # Get all endpoints
    endpoints = mapper.map_site("/", max_depth=2)
    
    # Scrape specific tags from each endpoint
    for endpoint in endpoints:
        # Scrape all paragraph texts
        paragraphs = scraper.scrape_by_tag("p", endpoint)
        scraper.save_to_file(paragraphs, f"paragraphs_{endpoint.replace('/', '_')}.txt")
        
        # Scrape by class
        content = scraper.scrape_by_class("content", endpoint)
        scraper.save_to_file(content, f"content_{endpoint.replace('/', '_')}.txt")
        
        # Scrape by ID
        main_content = scraper.scrape_by_id("main", endpoint)
        scraper.save_to_file(main_content, f"main_{endpoint.replace('/', '_')}.txt")
