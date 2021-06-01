import requests
#from firestore_interface import firestore_initialize
import firebase_admin
#from firebase_admin import credentials
from google.cloud import firestore
#import json



def get_github_device_codes():
    
    
    url = 'https://github.com/login/device/code'
    client_id = 'a5d9051591605e605159'
    scope = 'repo'
    data = {'client_id': client_id,
            'scope': scope}

    device_codes = requests.post(url=url, data = data)
    print(device_codes.status_code)
    print(device_codes.text)
    print(type(device_codes.text))
    
    return(device_codes.text)

def parse_request_response(text):
    "Takes the HTTP response text and parses it into a dict"

    # Find values with '='
    start_search_char = '='
    end_search_char = '&'
    result_dict = {}
    start_index = 0
    end_index = 0

    
    done = False
    while done == False:
        try:
            start_index = text.index(start_search_char)
    
            key_name = text[0:start_index]

            end_index = text.index(end_search_char)
            result = text[start_index+1:end_index]
            text = text[end_index+1:]
            
            result_dict[key_name] = result

            print(result_dict)    

        except ValueError as e:
            #print ("No such character available in string {}".format(text))
            done = True

    return result_dict

text = get_github_device_codes()
device_codes = parse_request_response(text)
verification_code = device_codes['user_code']
userid = '123imauserid'

default_app = firebase_admin.initialize_app()
db = firestore.Client()
users_ref = db.collection(u'exp').document(u'users')
doc_ref = users_ref.collection(userid).document(u'GitHub')
doc_ref.set({
            u'verification_code': verification_code
})



