# This script outputs the headlines of the top news stories from Google as of today

from urllib.request import Request, urlopen

#url = "https://fintel.io/ss/us/bngo"

keyword = 'subsidies'

# Prompt for ticker
print("Welcome to the Bloomberg News Headline Feed \n")
# Enable keyword versus free feed option
#stock = input("Enter a keyword to check top news: ")

url = 'https://www.bloomberg.com/'
# Add known browser user agent for site security check
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

#turn bytes to a string
type(webpage)
html = webpage.decode("utf-8")
print(html)

# Snip HTML to Top News
#top_news_html = 

# Snip HTML to Latest News
latest_html_start = html.find('<h3')
print("Latest: ", html[latest_html_start:latest_html_start+555])
print(latest_html_start)
# Looking for beginning of html headline:

start_index = html.find('headline_link">') 
#print(short_volume_index)
end_index = html.find("</a>")
print(end_index)
start_index = start_index + len('headline_link">') + 5
#start_index = end_index - 2
print(start_index)
print(html[start_index:start_index+20])

# End looking when see this:
end_index = html.find("</a>")
print("end index: ", (end_index))

total_results = html[start_index:end_index]
#print(f"The Short Volume Ratio is for {stock}: ", total_results, "percent")
#print(total_results)
