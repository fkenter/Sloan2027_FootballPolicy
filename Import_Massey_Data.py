import time
import pandas as pd
from io import StringIO
import bs4  # Use the bs4 library directly
import argparse # Import for handling command-line arguments
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# To run this script, you'll need to install a few packages:
# pip install beautifulsoup4 selenium webdriver-manager pandas

def get_soup(url: str) -> bs4.BeautifulSoup | None:
    """
    Fetches the HTML content of a given URL after JavaScript has loaded,
    and returns a BeautifulSoup object.

    Args:
        url: The URL of the website to scrape.

    Returns:
        A BeautifulSoup object representing the parsed HTML, or None if an
        error occurred.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run the browser in the background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = None  # Initialize driver to None
    try:
        # Set up the Selenium Chrome driver
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Wait for the main content container to be ready.
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, 'SHCtableleft'))
        )
        
        # Also wait for the table rows to be visible
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'tr.bodyrow'))
        )
        
        page_source = driver.page_source
        # Use bs4.BeautifulSoup to create the soup object
        soup = bs4.BeautifulSoup(page_source, 'html.parser')
        return soup

    except TimeoutException:
        print(f"Error: Timed out waiting for the page content to load for URL: {url}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for URL {url}: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def parse_html_table_to_dataframe(soup: bs4.BeautifulSoup) -> pd.DataFrame | None:
    """
    Parses a complex HTML table from the soup and returns a pandas DataFrame.
    This version is designed for the <tr>/<td> structure with nested 'detail' divs.

    Args:
        soup: The BeautifulSoup object containing the parsed page HTML.

    Returns:
        A pandas DataFrame with the ratings data, or None if parsing fails.
    """
    try:
        # Find all table rows with the class 'bodyrow'
        rows = soup.find_all('tr', class_='bodyrow')
        if not rows:
            print("Error: Could not find any table rows with class 'bodyrow'.")
            return None

        parsed_data = []
        for row in rows:
            cells = row.find_all('td')
            row_data = []

            # Cell 0: Team and Conference
            team_cell = cells[0]
            row_data.append(team_cell.find('a').get_text(strip=True) if team_cell.find('a') else '')
            row_data.append(team_cell.find('div', class_='detail').get_text(strip=True) if team_cell.find('div', class_='detail') else '')

            # Process all subsequent cells
            for cell in cells[1:]:
                main_text = ''
                for content in cell.contents:
                    if isinstance(content, bs4.NavigableString) and content.strip():
                        main_text = content.strip()
                        break
                detail_text = cell.find('div', class_='detail').get_text(strip=True) if cell.find('div', class_='detail') else ''
                row_data.append(main_text)
                row_data.append(detail_text)

            parsed_data.append(row_data)

        columns = [
            'Team', 'Conference', 'Record', 'Win Pct', 'Rank 1', 'Detail 1',
            'Rank 2', 'Detail 2', 'Rank 3', 'Detail 3', 'Rank 4', 'Detail 4',
            'Rank 5', 'Detail 5', 'Rank 6', 'Detail 6', 'Rank 7', 'Detail 7'
        ]
        
        df = pd.DataFrame(parsed_data)
        df.columns = columns[:len(df.columns)]
        return df

    except Exception as e:
        print(f"An error occurred during DataFrame parsing: {e}")
        return None


def Scrape_Massey_Ratings(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Scrapes Massey Ratings data for a given range of years and returns a single DataFrame.

    Args:
        start_year: The first year to scrape.
        end_year: The last year to scrape.

    Returns:
        A pandas DataFrame containing the combined data for all specified years.
        Returns an empty DataFrame if no data could be scraped.
    """
    all_years_data = [] # A list to hold the DataFrame for each year

    print(f"\nStarting to scrape data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        # Construct the URL for the current year
        target_url = f"https://masseyratings.com/cf{year}/fbs/ratings"
        print(f"\n--- Fetching data for {year} ---")
        print(f"URL: {target_url}")

        website_soup = get_soup(target_url)

        if website_soup:
            print(f"Successfully retrieved page source for {year}.")
            ratings_df = parse_html_table_to_dataframe(website_soup)

            if ratings_df is not None:
                # Add a 'Year' column to identify the season
                ratings_df['Year'] = year
                all_years_data.append(ratings_df)
                print(f"Successfully parsed data for {year}. Found {len(ratings_df)} teams.")
            else:
                print(f"Failed to create DataFrame for {year}.")
        else:
            print(f"Failed to retrieve and parse the website for {year}.")

    # Combine all the individual DataFrames into one
    if all_years_data:
        master_df = pd.concat(all_years_data, ignore_index=True)
        return master_df
    else:
        # Return an empty DataFrame if no data was collected
        return pd.DataFrame()
