# This script finds a list of majors

from urllib.request import Request, urlopen
import requests
import html
import json
import csv
import firebase_admin
from firebase_admin import credentials
from google.cloud import firestore


default_app = firebase_admin.initialize_app() 
url = f'http://www.act.org/content/act/en/research/reports/act-publications/college-choice-report-class-of-2013/college-majors-and-occupational-choices/college-majors-and-occupational-choices.html'
# Add known broswer user agent for site security check
req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

#turn bytes to a string
type(webpage)
html = webpage.decode("utf-8")

# ed_url = 'https://api.data.gov/ed/collegescorecard/v1/schools?api_key=zem3OldtbOzM55ZsUaUFATQgaY5ArfJvkuwl93wG'
# majors = requests.get(url).json()

# Looking for %:
start_index = html.find('NATURAL RESOURCES CONSERVATION</b></p>') 
#print(start_index)
end_index = html.find("Urban Affairs")
#print(end_index)
start_index = start_index + len('NATURAL RESOURCES CONSERVATION</b></p>') + 5
#start_index = end_index - 2
#print(start_index)

# End looking when see this:
end_index = html.find("Urban Affairs")
#print("end index: ", (end_index))

snippet = html[start_index:end_index]
#print("The List of Majors are ", snippet)
#print(type(snippet))

#Find categories

start_index = snippet.find('<p><b>') 
#print(start_index)
end_index = snippet.find('</p></b>')
#print(end_index)


categories = snippet[start_index:end_index]
print(categories)




# Now let's put each major as an item in a list of strings for Firebase
# Split by category 
#snippet = snippet.split('</b></p>')


# snippet = snippet.replace('</b></p>', '</li>')

# # Every major begins with <li> and ends with </li>

# # bad list

# bad_chars = [';', ':', '!', "*", "'", '"', "\n","<p><b>","<ul>","</ul>","&amp","</li>",'Area Studies, General (e.g., African, Middle Eastern)',',','/']

# # remove bad_chars
# for i in bad_chars :
#     snippet = snippet.replace(i, '')


# snippets = snippet.split('<li>')
# while("" in snippets) :
#     snippets.remove("")

# with open('list_of_majors','w') as f:
#     write = csv.writer(f)
#     write.writerow(snippets)


# # Add to firebase


#     # Add a new document
# db = firestore.Client()
# # doc_ref = db.collection(u'specializations').document(snippets[29])
# # doc_ref.set({
# #         u'name': snippets[29],
# #     })
# print(len(snippets))
# for i in range(0,len(snippets)):
#     doc_ref = db.collection(u'majors3').document(snippets[i])
#     doc_ref.set({
#             u'id': str(i),
#             u'name': snippets[i],
#         })
#     print(i)
#     print(snippets[i])

# # Then query for documents
# users_ref = db.collection(u'majors')

# for doc in users_ref.stream():
#     print(u'{} => {}'.format(doc.id, doc.to_dict()))



