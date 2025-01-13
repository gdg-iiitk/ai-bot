from bs4 import BeautifulSoup
import requests
import os
import time
from urllib.parse import urljoin, urlparse, parse_qs, urlsplit
from ratelimit import limits, sleep_and_retry
from requests.exceptions import RequestException
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from functools import lru_cache

class WebScraper:
    def __init__(self, base_url, rate_limit=1):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit = rate_limit
        self.priority_tags = [
            'h1', 'h2', 'h3', 'title',
            'p', 'article', 'section',
            'td', 'th', 'li'
        ]
        self.excluded_patterns = {
            # File extensions to skip
            r'\.(?:css|js|json|xml|jpg|jpeg|png|gif|pdf|ico|woff|woff2|ttf|eot)$',
            # External services
            r'(?:youtube|twitter|x\.com|facebook|linkedin|instagram)',
            # Common API and asset paths
            r'(?:api|assets|static|cdn|media|images)/',
            # Payment and external systems
            r'(?:payment|sbi|collect|gateway)',
            # Auth and utility paths
            r'(?:login|logout|auth|signup|register)'
        }
        self.data_dir = "data/content"
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Storage directory created at: {self.data_dir}")
        self.visited_urls = set()
        self.excluded_domains = ['youtube.com', 'twitter.com', 'facebook.com', 
                               'linkedin.com', 'instagram.com']  # Add this line
        self.debug = True  # Add debug mode
        
        # Setup Selenium
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(2)  # Reduced to 2 seconds
        self.max_workers = 12  # Increased parallel processing
        self.cache = {}  # Simple cache
        
        # Update institutional content selectors with more specific targets
        self.content_selectors = {
            'home': [
                '.home-content', 
                '#home-banner',
                '.institute-highlights',
                '.news-updates',
                '.announcements'
            ],
            'about': [
                '#about-institute',
                '.about-content',
                '.vision-mission',
                '.director-message'
            ],
            'academics': [
                '#academic-programs',
                '.course-details',
                '.program-structure',
                '.curriculum'
            ],
            'research': [
                '#research-areas',
                '.research-highlights',
                '.publications',
                '.projects',
                '.laboratories'
            ],
            'faculty': [
                '#faculty-list',
                '.faculty-profile',
                '.department-faculty',
                '.faculty-research'
            ],
            'admissions': [
                '#admission-process',
                '.eligibility-criteria',
                '.how-to-apply',
                '.important-dates'
            ]
        }
        
        # Priority sections to scrape first
        self.priority_sections = ['home', 'about', 'research', 'academics', 'faculty']
        
        # Add specific faculty selectors
        self.faculty_selectors = {
            'faculty_profile': [
                '.faculty-profile',
                '.faculty-info',
                '.faculty-details',
                '.profile-card',
                '#faculty-list .member',
                '.faculty-member',
                '[class*="faculty"]',  # Match any class containing "faculty"
                '.profile-container'
            ],
            'faculty_sections': [
                'name',
                'designation',
                'department',
                'qualification',
                'specialization',
                'research_interests',
                'email',
                'phone',
                'website'
            ]
        }
        
        self.text_bearing_tags = [
            'p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'li', 'td', 'th', 'a', 'label', 'strong', 'em', 'i', 'b',
            'article', 'section', 'main', 'aside', 'footer', 'header',
            'blockquote', 'cite', 'pre', 'code', 'small', 'time',
            'address', 'figcaption', 'dt', 'dd'
        ]
        
        # Words that indicate meaningful content
        self.content_indicators = [
            'professor', 'faculty', 'research', 'interests', 'specialization',
            'education', 'qualification', 'experience', 'publications', 'projects',
            'teaching', 'courses', 'expertise', 'department', 'contact', 'email',
            'phone', 'office', 'lab', 'group', 'awards', 'recognition'
        ]

    def log(self, message):
        """Debug logging helper"""
        if self.debug:
            print(f"[DEBUG] {message}")

    @sleep_and_retry
    @limits(calls=3, period=1)  # Increased rate limit to 3 calls per second
    def get_soup(self, endpoint, timeout=2):  # Reduced timeout from 3 to 2 seconds
        """Enhanced get_soup with better hashbang handling"""
        url = urljoin(self.base_url, endpoint)
        self.log(f"Fetching URL: {url}")
        
        try:
            # Handle hashbang URLs
            if '#!' in url:
                base_url = url.split('#!')[0]
                hashbang_path = url.split('#!')[1]
                self.log(f"Hashbang detected - Base: {base_url}, Path: {hashbang_path}")
                
                # Try both the full URL and the base URL
                response = self.session.get(url, timeout=timeout)
                if response.status_code != 200:
                    self.log("Trying base URL instead of hashbang URL")
                    response = self.session.get(base_url, timeout=timeout)
            else:
                response = self.session.get(url, timeout=timeout)
            
            response.raise_for_status()
            self.log(f"Response status: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            self.log(f"Successfully created soup object for {url}")
            return soup
            
        except Exception as e:
            self.log(f"Error fetching {url}: {str(e)}")
            return None

    def is_valid_content_url(self, url):
        """Check if URL is valid for content scraping"""
        if not url or not isinstance(url, str):
            return False
            
        # Check against excluded patterns
        for pattern in self.excluded_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
                
        parsed = urlparse(url)
        return (
            parsed.netloc == urlparse(self.base_url).netloc and
            not any(url.startswith(x) for x in [
                'javascript:', 'mailto:', 'tel:', 'data:', 
                'about:', 'file:', '#', '{', '%7B'
            ])
        )

    def clean_url(self, url):
        """Clean and normalize URL paths"""
        if not url:
            return ''
        
        # Handle absolute URLs
        if url.startswith(self.base_url):
            url = url[len(self.base_url):]
            
        # Remove query parameters and fragments
        url = urlsplit(url).path
        
        # Normalize path
        url = url.rstrip('/')
        if not url:
            url = '/'
            
        return url

    @lru_cache(maxsize=100)
    def get_dynamic_soup(self, endpoint, wait_time=2):  # Reduced to 2 seconds
        """Get soup from dynamically rendered page with caching"""
        if endpoint in self.cache:
            return self.cache[endpoint]
            
        url = urljoin(self.base_url, endpoint)
        self.log(f"Loading dynamic content from: {url}")
        
        try:
            self.driver.get(url)
            # Minimum wait for content
            time.sleep(1)
            
            # Quick check for content
            try:
                WebDriverWait(self.driver, 2).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )
            except TimeoutException:
                pass
            
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            self.cache[endpoint] = soup
            return soup
            
        except Exception as e:
            self.log(f"Error loading dynamic content: {str(e)}")
            return None

    def scrape_all_readable(self, endpoint):
        """Enhanced content extraction for dynamic pages"""
        self.log(f"Starting dynamic content extraction for: {endpoint}")
        
        # Use Selenium instead of requests
        soup = self.get_dynamic_soup(endpoint)
        if not soup:
            return None

        content = []
        
        # Extract title
        title = soup.find('title')
        if title:
            content.append({
                'tag': 'title',
                'text': title.text.strip()
            })

        # Extract content based on institutional selectors
        for section, selectors in self.content_selectors.items():
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    self.log(f"Found {section} section with selector {selector}")
                    for element in elements:
                        # Extract headings within the section
                        headings = element.find_all(['h1', 'h2', 'h3', 'h4'])
                        for heading in headings:
                            content.append({
                                'tag': heading.name,
                                'text': heading.get_text(strip=True)
                            })
                        
                        # Extract paragraphs within the section
                        paragraphs = element.find_all('p')
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if len(text) > 20:  # Only meaningful paragraphs
                                content.append({
                                    'tag': 'p',
                                    'text': text
                                })

        # Extract lists and tables
        for tag in ['ul', 'ol', 'table']:
            elements = soup.find_all(tag)
            for element in elements:
                items = element.find_all(['li', 'tr'])
                for item in items:
                    text = item.get_text(strip=True)
                    if len(text) > 10:
                        content.append({
                            'tag': item.name,
                            'text': text
                        })

        self.log(f"Extracted {len(content)} content items")
        return content if content else None

    def scrape_priority_content(self):
        """Scrape priority sections in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_section = {
                executor.submit(self.scrape_section, section): section
                for section in self.priority_sections
            }
            
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    content = future.result()
                    if content:
                        self.save_content(section, content)
                except Exception as e:
                    self.log(f"Error processing {section}: {str(e)}")

    def scrape_section(self, section):
        """Scrape individual section"""
        endpoint = f"/#!/{section}"
        return self.scrape_all_readable(endpoint)

    def save_content(self, section, content):
        """Save section content to file"""
        try:
            safe_name = f"{section.lower()}"
            output_file = os.path.join(self.data_dir, f"{safe_name}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"URL: {urljoin(self.base_url, f'/#!/{section}')}\n")
                f.write(f"Section: {section.upper()}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                for item in content:
                    f.write(f"[{item['tag'].upper()}] {item['text']}\n")
                    f.write("="*80 + "\n")
                    
        except Exception as e:
            self.log(f"Error saving {section}: {str(e)}")

    def __del__(self):
        """Clean up Selenium driver"""
        if hasattr(self, 'driver'):
            self.driver.quit()

    def is_meaningful_text(self, text):
        """Check if text content is meaningful"""
        if not text or len(text.strip()) < 3:
            return False
            
        text = text.lower()
        
        # Ignore navigation and common UI elements
        ignore_phrases = ['next', 'previous', 'menu', 'close', 'open', 'click', 'loading']
        if any(phrase in text for phrase in ignore_phrases):
            return False
            
        # Check if text contains meaningful indicators
        return any(indicator in text for indicator in self.content_indicators)

    def extract_all_text(self, element, depth=0):
        """Recursively extract all text from element and its children"""
        texts = []
        
        if hasattr(element, 'name') and element.name in self.text_bearing_tags:
            # Get direct text from this element
            direct_text = element.string
            if direct_text and direct_text.strip():
                text = direct_text.strip()
                if self.is_meaningful_text(text):
                    texts.append({
                        'tag': element.name,
                        'text': text,
                        'depth': depth
                    })
        
        # Recursively process child elements
        for child in element.children:
            if hasattr(child, 'children'):
                child_texts = self.extract_all_text(child, depth + 1)
                texts.extend(child_texts)
        
        return texts

    def extract_faculty_data(self, element):
        """Extract all possible faculty information"""
        data = {}
        
        # Extract all text content recursively
        all_texts = self.extract_all_text(element)
        
        # Group related information
        for item in all_texts:
            text = item['text'].strip()
            lower_text = text.lower()
            
            # Categorize information
            if any(title in lower_text for title in ['dr.', 'prof.', 'professor']):
                data['name'] = text
            elif any(role in lower_text for role in ['professor', 'assistant', 'associate', 'head']):
                data['designation'] = text
            elif any(qual in lower_text for qual in ['phd', 'ph.d', 'm.tech', 'b.tech']):
                data.setdefault('qualifications', []).append(text)
            elif '@' in text:
                data['email'] = text
            elif 'research' in lower_text or 'interests' in lower_text:
                data.setdefault('research_interests', []).append(text)
            elif 'specialization' in lower_text:
                data.setdefault('specializations', []).append(text)
            elif any(word in lower_text for word in ['publication', 'journal', 'conference']):
                data.setdefault('publications', []).append(text)
            else:
                data.setdefault('additional_info', []).append(text)
        
        return data

    def save_faculty_data(self, faculty_data):
        """Save faculty data with better formatting"""
        output_file = os.path.join(self.data_dir, "faculty.txt")
        
        # Create backup of existing file
        if os.path.exists(output_file):
            backup_file = output_file + '.bak'
            try:
                os.replace(output_file, backup_file)
                self.log(f"Created backup at: {backup_file}")
            except Exception as e:
                self.log(f"Error creating backup: {str(e)}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Faculty Directory - IIIT Kottayam\n")
                f.write(f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Faculty Members: {len(faculty_data)}\n\n")
                
                for i, profile in enumerate(faculty_data, 1):
                    f.write(f"\nFaculty Member #{i}\n")
                    f.write("-" * 50 + "\n")
                    
                    # Write structured data first
                    priority_fields = ['name', 'designation', 'email']
                    for field in priority_fields:
                        if field in profile:
                            f.write(f"{field.replace('_', ' ').title()}: {profile[field]}\n")
                    
                    # Write list items
                    list_fields = ['qualifications', 'research_interests', 'specializations', 'publications']
                    for field in list_fields:
                        if field in profile and profile[field]:
                            f.write(f"\n{field.replace('_', ' ').title()}:\n")
                            for item in profile[field]:
                                f.write(f"- {item}\n")
                    
                    if 'additional_info' in profile and profile['additional_info']:
                        f.write("\nAdditional Information:\n")
                        for info in profile['additional_info']:
                            f.write(f"- {info}\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
                
            self.log(f"Successfully saved faculty data to: {output_file}")
            
        except Exception as e:
            self.log(f"Error saving faculty data: {str(e)}")

    def scrape_faculty_content(self):
        """Scrape faculty data in parallel"""
        self.log("Starting faculty data extraction...")
        endpoint = "/#!/faculty"
        
        soup = self.get_dynamic_soup(endpoint)
        if not soup:
            return None

        faculty_elements = []
        for selector in self.faculty_selectors['faculty_profile']:
            elements = soup.select(selector)
            if elements:
                faculty_elements.extend(elements)
                self.log(f"Found {len(elements)} faculty profiles with selector: {selector}")

        # Process faculty data in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            faculty_data = list(executor.map(self.extract_faculty_data, faculty_elements))

        # Filter out empty entries
        faculty_data = [data for data in faculty_data if data]
        
        # Save faculty data
        if faculty_data:
            self.save_faculty_data(faculty_data)
        
        return faculty_data

class SiteMapper:
    def __init__(self, scraper):
        self.scraper = scraper
        self.visited = set()
        self.domain = urlparse(scraper.base_url).netloc
        self.priority_paths = [
            r'^/$',  # Homepage
            r'/about',
            r'/academics',
            r'/admission',
            r'/department',
            r'/faculty',
            r'/research',
            r'/placement',
            r'/programmes',
            r'/infrastructure'
        ]
        self.hashbang_paths = set()  # Add this line to initialize hashbang_paths

    def extract_links(self, soup):
        if not soup:
            return set()
        
        links = set()
        for link in soup.find_all('a', href=True):
            url = link['href'].strip()
            if self.scraper.is_valid_content_url(url):
                full_url = urljoin(self.scraper.base_url, url)
                path = urlparse(full_url).path
                if any(re.search(pattern, path) for pattern in self.priority_paths):
                    links.add(full_url)
        return links

    def is_valid_url(self, url):
        parsed = urlparse(url)
        if any(domain in parsed.netloc.lower() for domain in self.scraper.excluded_domains):
            return False
        # Handle both regular and hashbang URLs
        if '#!' in url:
            base_url = url.split('#!')[0]
            return urlparse(base_url).netloc == self.domain
        return parsed.netloc == self.domain

    def clean_endpoint(self, url):
        """Preserve hashbang paths while cleaning URLs"""
        if '#!' in url:
            # Keep the hashbang path intact
            if url.startswith(self.scraper.base_url):
                _, path = url.split(self.scraper.base_url, 1)
            else:
                path = url
            return path
        return self.scraper.clean_url(url)  # Now using the WebScraper's clean_url method

    def extract_links(self, soup):
        links = set()
        if not soup:
            return links

        # Extract from various sources
        sources = [
            ('a', 'href'),
            ('link', 'href'),
            ('script', 'src'),
            ('img', 'src'),
            ('form', 'action')
        ]

        for tag, attr in sources:
            for element in soup.find_all(tag, {attr: True}):
                url = element[attr].strip()
                if self.is_valid_link(url):
                    if '#!' in url or '#!/' in url:
                        self.hashbang_paths.add(url)
                    else:
                        full_url = urljoin(self.scraper.base_url, url)
                        links.add(full_url)

        # Extract URLs from JavaScript
        scripts = soup.find_all('script', string=True)
        for script in scripts:
            urls = re.findall(r'["\'](/[^"\']*)["\']', script.string or '')
            for url in urls:
                if self.is_valid_link(url):
                    full_url = urljoin(self.scraper.base_url, url)
                    links.add(full_url)

        return links

    def is_valid_link(self, url):
        if not url:
            return False
        return not any(url.startswith(x) for x in [
            'javascript:', 'mailto:', 'tel:', 'data:', 
            'about:', 'file:', '#', '{', '%7B'
        ])

    def is_priority_path(self, path):
        return any(re.search(pattern, path) for pattern in self.priority_patterns)

    def map_site(self, start_path, max_depth=2):
        clean_path = self.clean_endpoint(start_path)
        if max_depth <= 0 or clean_path in self.visited:
            return set()
        
        self.visited.add(clean_path)
        soup = self.scraper.get_soup(clean_path)
        endpoints = {clean_path}
        
        # Extract and process links
        links = self.extract_links(soup)
        for link in links:
            if '#!' in link or '#!/' in link:
                self.hashbang_paths.add(link)
            else:
                path = self.clean_endpoint(link)
                if path and path not in self.visited:
                    endpoints.update(self.map_site(path, max_depth - 1))
        
        # Ensure hashbang paths are included
        endpoints.update(self.hashbang_paths)
        return endpoints

def scrape_full_site(base_url, max_depth=2):
    scraper = WebScraper(base_url)
    
    # Scrape priority content in parallel
    print("Scraping priority sections...")
    scraper.scrape_priority_content()
    
    # Process remaining routes in larger parallel batches
    remaining_routes = [
        '/#!/programmes',
        '/#!/department',
        '/#!/placement',
        '/#!/infrastructure',
        '/#!/contact'
    ]

    print("Processing remaining routes...")
    with ThreadPoolExecutor(max_workers=12) as executor:  # Increased workers
        futures = [
            executor.submit(process_endpoint, scraper, base_url, endpoint)
            for endpoint in remaining_routes
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")

    print("Scraping completed!")

def process_endpoint(scraper, base_url, endpoint):
    """Enhanced endpoint processing with better hashbang handling"""
    try:
        scraper.log(f"\nProcessing endpoint: {endpoint}")
        
        # Improved hashbang URL handling
        if '#!' in endpoint:
            path_parts = endpoint.split('#!')[1].strip('/').split('/')
            safe_endpoint = '_'.join(path_parts) or 'home'
            scraper.log(f"Hashbang path parts: {path_parts}")
        else:
            path_parts = endpoint.strip('/').split('/')
            safe_endpoint = '_'.join(path_parts) or 'home'
            
        scraper.log(f"Safe endpoint name: {safe_endpoint}")
        output_file = os.path.join(scraper.data_dir, f"{safe_endpoint}.txt")
        
        content = scraper.scrape_all_readable(endpoint)
        if content:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"URL: {urljoin(base_url, endpoint)}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                for item in content:
                    f.write(f"[{item['tag'].upper()}] {item['text']}\n")
                    f.write("="*80 + "\n")
            scraper.log(f"Successfully saved content to: {output_file}")
        else:
            scraper.log(f"No content found for: {endpoint}")
            
    except Exception as e:
        scraper.log(f"Error processing {endpoint}: {str(e)}")
        import traceback
        scraper.log(traceback.format_exc())

# Fix the BASE_URL syntax
BASE_URL = "https://iiitkottayam.ac.in"  # Remove the hashbang from base URL

# Run with optimized settings
if __name__ == "__main__":
    BASE_URL = "https://iiitkottayam.ac.in"
    scrape_full_site(BASE_URL, max_depth=4)  # Reduced depth for speed

def main():
    BASE_URL = "https://iiitkottayam.ac.in"
    scraper = WebScraper(BASE_URL)
    faculty_data = scraper.scrape_faculty_content()
    if faculty_data:
        scraper.save_faculty_data(faculty_data)

if __name__ == "__main__":
    main()