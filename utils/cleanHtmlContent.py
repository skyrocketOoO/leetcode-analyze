from bs4 import BeautifulSoup

def CleanHtmlContent(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    cleaned_text = soup.get_text(separator=" ")  # Extract plain text and separate blocks by space
    return cleaned_text
