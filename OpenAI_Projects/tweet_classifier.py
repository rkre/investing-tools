
import openai
from secrets import OPENAI_API_TOKEN

#openai.api_key = {OPENAI_API_TOKEN}
openai.api_key = "sk-IGR8QQSTG3iOLZQyYYC0SAuavVvuMy0Pc4b1eXDw"

response = openai.Completion.create(
  engine="davinci",
  prompt="This is a tweet sentiment classifier\nTweet: \"Loaded 25,000 $SAVA Friday plan possible 75,000 more next week.\"\nSentiment: Positive\n###\nTweet: \"I was in the $AMC $SAVA and $FUBO runs and sold after the curl down.\"\nSentiment: Negative\n###\nTweet: \"$SAVA HC Wainwright & Co. Maintains Buy on Cassava Sciences, Raises Price Target to $66\"\nSentiment: Positive\n###\nTweet: \"This is the link to the article\"\nSentiment: Neutral\n###\nTweet text\n\n\n1. \"I loved the new Batman movie!\"\n2. \"I hate it when my phone battery dies\"\n3. \"My day has been 👍\"\n4. \"This is the link to the article\"\n5. \"This new music video blew my mind\"\n\n\nTweet sentiment ratings:\n1: Positive\n2: Negative\n3: Positive\n4: Neutral\n5: Positive\n\n\n###\nTweet text\n\n\n1. \"I can't stand homework\"\n2. \"This sucks. I'm bored 😠\"\n3. \"I can't wait for Halloween sucks!!!\"\n4. \"$SAVA added 10k shares here (swing idea )\"\n5. \"I hate chocolate\"\n\n\nTweet sentiment ratings:\n1.",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"]
)

print(response)