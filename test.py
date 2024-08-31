import firebase_admin
from firebase_admin import firestore


app=firebase_admin.initialize_app(firebase_admin.credentials.Certificate('./foriegn-traveller-firebase-adminsdk-uvg5y-cb35de410f.json'))

db=firestore.client()


stores=list(map(lambda x:x.to_dict(),db.collection('stores').stream()))
        
for s in stores:
    print(s)
    print('===============')
