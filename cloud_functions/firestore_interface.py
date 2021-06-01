
#from firebase import Firebase

import firebase_admin
from firebase_admin import credentials
import requests
from google.cloud import firestore
import json




def firestore_initialize():
    "Initializing firebase. Note: need to set environment paths to json from firebase"
    default_app = firebase_admin.initialize_app()
    db = firestore.Client()
    return db


def firestore_add_doc(collection_name, data_list, doc_name, field_names, db):
    "Add a new document"

    db = firestore_initialize

    for i in range(len(data_list)):
        doc_ref = db.collection(collection_name).document(data_list[i][doc_name]) 
        for j in range(len(field_names)):
            doc_ref.set({
                field_names[i]: data_list[i][field_names[i]],

            })
            print("Wrote field name: ", field_names[i])

    print("Successfully wrote data to Firestore!")

def firestore_read_docs(db, collection_name):
    # Then query for documents
    users_ref = db.collection(u'testuniversities')

    for doc in users_ref.stream():
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    
