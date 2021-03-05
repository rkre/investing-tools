import os
import openai

#openai.api_key = os.environ["OPENAI_API_KEY"]




response = openai.Completion.create(
  engine="davinci",
  prompt="Social media post: \"The new episode of The Mandalorian was great\"\nSentiment (positive, neutral, negative):",
  temperature=0,
  max_tokens=1,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)