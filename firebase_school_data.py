#from firebase import Firebase

import firebase_admin
from firebase_admin import credentials
import requests
from google.cloud import firestore
import json

url = "https://universities-and-colleges.p.rapidapi.com/universities"


default_app = firebase_admin.initialize_app()


headers = {
    'x-rapidapi-key': "017c5e2659msh582cb5f1afa0068p172a78jsna5005d1fcd27",
    'x-rapidapi-host': "universities-and-colleges.p.rapidapi.com"
    }

limit = "50"
pages = 1

def get_initials(name):
    "Returns the initials of a string for names"
    initials = []
    initials.append(name[0])
    space_counter = 1

    for i in range(0,len(name)):
        
        if name[i] == ' ':
            initials.append(name[i+1])
    
    print(initials)
    return initials

def fill_empty_data(school_list):

    for i in range(len(school_list)):

        # if 'id' in school_list:
        #     continue
        #     #print("\nid exist for this school\n")
        # else:
        #     school_list[i]['id'] = 'NA'
            

        # if 'name' in school_list:
        #     continue
        #     #print("\nname exist for this school\n")
        # else:
        #     school_list[i]['name'] = 'NA'


        if 'colors' in school_list[i]:
            print("\ncolors exist for this school\n")
        else:
            school_list[i]['colors'] = 'NA'
            
        
        if 'type' in school_list[i]:
            print("\ntype exist for this school\n")
        else:
            school_list[i]['type'] = 'NA'

        if 'atheleticNickname' in school_list[i]:
            print("\natheletic nickname exist for this school\n")
        else:
            school_list[i]['atheleticNickname'] = 'NA'

        if 'mascot' in school_list[i]:
            print("\nmascot exist for this school\n")
        else:
            school_list[i]['mascot'] = 'NA'
            
        
        if 'address' in school_list[i]:
            print("\naddress exist for this school\n")
        else:
            school_list[i]['address'] = 'NA'
        
        if 'city' in school_list[i]:
            print("\ncity exist for this school\n")
        else:
            school_list[i]['city'] = 'NA'
            
        
        if 'state' in school_list[i]:
            print("\ntype exist for this school\n")
        else:
            school_list[i]['state'] = 'NA'

        if 'stateCode' in school_list[i]:
            print("\ntstateCode exist for this school\n")
        else:
            school_list[i]['stateCode'] = 'NA'
            
        
        if 'zip' in school_list[i]:
            print("\nzip exist for this school\n")
        else:
            school_list[i]['zip'] = 'NA'
        
        if 'country' in school_list[i]:
            print("\ncountry exist for this school\n")
        else:
            school_list[i]['country'] = 'NA'
            
        
        if 'countryCode' in school_list[i]:
            print("\ntcountry code exist for this school\n")
        else:
            school_list[i]['countryCode'] = 'NA'
            
        
        if 'website' in school_list[i]:
            print("\nwebsite exist for this school\n")
        else:
            school_list[i]['website'] = 'NA'
            

    print(school_list)
    print(i)
    return school_list
    
def firebase_ref():
    # Add a new document
    db = firestore.Client()
    return db



def store_in_firebase(school_list, db):
    

    for school in range(len(school_list)):
        
        # Name collections and documents 
        collection_name = ''
        initials = get_initials(school_list[school]['name'])
        collection_name = collection_name.join(initials)

        doc_name = collection_name + '_info'

        #doc_ref = db.collection(u'universities').document(school_list[school]['name'])


        school_ref = db.collection(u'testestestest').document(u'universities')
        doc_ref = school_ref.collection(collection_name).document(doc_name)


        doc_ref.set({
            u'id': school_list[school]['id'],
            u'name': school_list[school]['name'],
            u'colors': school_list[school]['colors'],
            u'type' : school_list[school]['type'],
            u'atheleticNickname': school_list[school]['atheleticNickname'],
            u'mascot': school_list[school]['mascot'],
            u'address': school_list[school]['address'],
            u'city' : school_list[school]['city'],
            u'state': school_list[school]['state'],
            u'stateCode': school_list[school]['stateCode'],
            u'zip': school_list[school]['zip'],
            u'country' : school_list[school]['country'],
            u'countryCode': school_list[school]['countryCode'],
            u'website' : school_list[school]['website'],

        })

        print('Stored!')
        

def load_from_firebase(db, db_name):
    # Then query for documents
    users_ref = db.collection(db_name)

    for doc in users_ref.stream():
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    
    print('Done!')

def get_school_list(page):
    
    querystring = {"includeUniversityDetails":"true", "countryCode":"US", "limit": limit, "page": str(page)}

    school_list = requests.request("GET", url, headers=headers, params=querystring).json()
    return school_list

def get_schools():

    for page in range(0,pages):

        school_list = get_school_list(page)

        print(page)

        school_list = fill_empty_data(school_list)

        #cred = credentials.Certificate("json_keys/test-58ca2-firebase-adminsdk-dpkr3-0255f2ddea.json")


        #firebase_admin.initialize_app(cred)
        db = firebase_ref()
        store_in_firebase(school_list, db)
        load_from_firebase(db)

    
    

    

get_schools()






# config = {
#   "apiKey": "apiKeyAIzaSyCEUr0-I6gB2_uHWiberAXYsFAHR1Np7-Q",
#   "authDomain": "test-58ca2.firebaseapp.com",
#   "databaseURL": "https://test-58ca2.firebaseio.com",
#   "storageBucket": "test-58ca2.appspot.com"
# }

# firebase = Firebase(config)

# # Get a reference to the auth service
# auth = firebase.auth()

# # Log the user in
# user = auth.sign_in_with_email_and_password('rkreynin@gmail.com', 'p6345650')

# # Get a reference to the database service
# db = firebase.database()

# # data to save
# data = {
#     "name": "Joe Tilsed"
# }

# # Pass the user's idToken to the push method
# results = db.child("users").push(data, user['idToken'])

# data = {"name": "Joe Tilsed"}
# db.child("users").child("Joe").set(data)