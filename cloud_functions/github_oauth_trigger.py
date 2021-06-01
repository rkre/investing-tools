def hello_firestore(event, context):
    """Triggered by a change to a Firestore document.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    resource_string = context.resource
    # print out the resource string that triggered the function
    print(f"Function triggered by change to: {resource_string}.")
    # now print out the entire event object

    import requests
    url = 'https://github.com/login/device/code'
    client_id = 'a5d9051591605e605159'
    scope = 'repo'
    data = {'client_id': client_id,
            'scope': scope}     

    device_codes = requests.post(url=url, data = data)
    print(device_codes.status_code)


    text = device_codes.text
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
            print ("Done!")
            done = True


    userid = '123imauserid'

    import firebase_admin
    from google.cloud import firestore
    default_app = firebase_admin.initialize_app()
    db = firestore.Client()
    users_ref = db.collection(u'exp').document(u'users')
    doc_ref = users_ref.collection(userid).document(u'GitHub')
    doc_ref.set({
            u'verification_code': verification_code
    })


    print(str(event))
