import openai
import secrets
from urllib.request import Request, urlopen

#response = openai.Completion.create(engine="davinci", prompt="My cat likes to", max_tokens=10)
openai.api_key = 'sk-IGR8QQSTG3iOLZQyYYC0SAuavVvuMy0Pc4b1eXDw'
#openai.api_key = os.environ["OPENAI_API_KEY"]

url = f'https://investors.beamtx.com/node/7006'
# Add known broswer user agent for site security check
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()
#turn bytes to a string
type(webpage)
html = webpage.decode("utf-8")
#print(html)
start_index = html.find("<h3 class=") + len('<h3 class=') + 14
end_index = html.find("</article>") - 100
news_string = html[start_index:end_index] + "STOP\n\ntl;dr"
print(news_string)
# News string needs to be limited in length
#news_string = html + "\n\ntl:dr"
#news_string = "CAMBRIDGE, Mass., Feb. 23, 2021 (GLOBE NEWSWIRE) -- Beam Therapeutics Inc. (Nasdaq: BEAM), a biotechnology company developing precision genetic medicines through base editing, today announced it has completed the acquisition of Guide Therapeutics, Inc. (“GuideTx”), a developer of nonviral drug delivery vehicles for genetic medicines, further expanding the potential reach of Beam’s genetic medicines into new target tissues and diseases." + "STOP\n\ntl:dr"

response = openai.Completion.create(
  engine="davinci",
  prompt=news_string,
  temperature=0.3,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["STOP"]
)

print(response)
