# This script checks the Short Sale Volume Ratio based on the fintel website

from urllib.request import Request, urlopen

#url = "https://fintel.io/ss/us/bngo"

def short_seller_checker(ticker):

    # Prompt for ticker
    #print("Welcome to the Short Seller Checker \n")
    #stock = input("Enter the ticker to check short seller ratio: ")
    stock = ticker
    url = f'https://fintel.io/ss/us/{stock}'
    # Add known broswer user agent for site security check
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    #turn bytes to a string
    type(webpage)
    html = webpage.decode("utf-8")


    # Looking for %:
    short_volume_index = html.find('Short Volume Ratio</td>') 
    #print(short_volume_index)
    end_index = html.find("%</td>")
    #print(end_index)
    start_index = short_volume_index + len('Short Volume Ratio</td>') + 5
    #start_index = end_index - 2
    #print(start_index)

    # End looking when see this:
    end_index = html.find("%</td>")
    #print("end index: ", (end_index))

    total_results = html[start_index:end_index]
    print(f"The Short Volume Ratio is for {stock}: ", total_results, "percent")

    #convert to int
    #total_int = int(total_results)
    #print(total_int)

    #return total_results

