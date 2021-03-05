# Outputs a quick synposis on a ticker of choice

from urllib.request import Request, urlopen



def website(ticker):
    "Checks for website"
    #Scrapes google search

    #url1 = 'https://news.google.com/topstories?hl=en-US&gl=US&ceid=US:en'
    url = f'https://www.google.com/search?q={ticker}'
    # Add known browser user agent for site security check
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    #turn bytes to a string
    type(webpage)
    html = webpage.decode("utf-8")
    print(html)

    # Snip HTML for website
    snip_start = html.find('<div class="yuRUbf"><a href=')
    print("Website ", html[snip_start:snip_start+20])

    # End snip for top news


