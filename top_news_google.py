# This script outputs the headlines of the top news stories from Google as of today

from urllib.request import Request, urlopen

#url = "https://fintel.io/ss/us/bngo"

keyword = 'subsidies'

# Prompt for ticker
print("Welcome to the Google News Headline Feed \n")
# Enable keyword versus free feed option
stock = input("Enter a keyword to check top news: ")

url1 = 'https://news.google.com/topstories?hl=en-US&gl=US&ceid=US:en'
url2 = f'https://news.google.com/search?q={keyword}&hl=en-US&gl=US&ceid=US%3Aen'
# Add known browser user agent for site security check
req = Request(url1, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

#turn bytes to a string
type(webpage)
html = webpage.decode("utf-8")
print(html)

# Snip HTML for top news
top_news_snip_start = html.find('"DY5T1d RZIKme" >')
print("Top News: ", html[top_news_snip_start:top_news_snip_start+20])

# End snip for top news

# Looking for %:


