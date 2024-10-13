from flask_restful import Resource, reqparse,abort
from flask import request, Response, jsonify,current_app,make_response
import pymongo
from joblib import load
from PIL import Image
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
from firebase_admin import firestore
from dotenv import load_dotenv
import random
import requests
import json
from openai import OpenAI

load_dotenv()
class SaveUserController(Resource):

    def __init__(self):
        self.db=current_app.config['DATABASE']
        self.database = self.db.get_database('users')
        self.collection = self.database.get_collection('users')
    
    def get(self):
        
        # model=joblib.load('models/user_cat.joblib')

        return jsonify({"msg": "hello world"})

        

    def post(self):
        if  not request.is_json:
            return Response("Not in valid JSON format", status=400)

        try:
            
            data = dict(request.get_json())

            if 'email' not in data.keys() or 'userid' not in data.keys():
                print('aborting')
                return Response("Email and User ID are required fields", status=400) 

            

            
            query = {"$or": [{"email": data['email']}, {"userid": data['userid']}]}
            # print("come to try")
            user = self.collection.find_one(query, limit=1)

            if user:
                return Response("user already exists!",status=400)  # User already exists

            # Create document with only required fields
            doc = {
                "userid": data['userid'],
                "email": data['email'],
                "preferences": {
                    "Climate_preference": '',
                    "Preferred_cuisines": '',
                    "Specific_food_categories": '',
                    "Preferred_activities": '',
                    "Travel_Preferences": '',
                    "Budget": '',
                    "County":'',
                },
                "personal_data": {
                    "age": 0,
                    "gender": '',
                    "married": '',
                    "graduated": '',
                    "job": '',
                    "family_size": 1,
                    "spending":'',
                }
            }

            new_user = self.collection.insert_one(doc)
            print(new_user)
            return Response("Created", status=201)
        
        
        except pymongo.errors.DuplicateKeyError as e:
            return Response({"error": "User already exists ","msg":str(e)}, status=409)  # More specific error message
        except Exception as e:
            # Handle other exceptions with appropriate error codes and messages
            print('came here 500')
            
            return Response("Internal Server Error", status=500)
        
    
    def put(self):
        req=reqparse.RequestParser()
        req.add_argument('userid',type=str,required=True,help="User ID is required",
                        location='json')
        
        req.add_argument('p',type=str,required=True,help="what is to update is required!",location='json')

        
        req.add_argument('Climate_preference',type=str,help="Climate preference is required",
                        location='json')
        req.add_argument('Preferred_cuisines',type=str,help="Preferred cuisines is required",location='json')
        req.add_argument('Specific_food_categories',type=str,help="Specific food categories is required",location='json')
        req.add_argument('Preferred_activities',type=str,help="Preferred activities is required",location='json')
        req.add_argument('Travel_Preferences',type=str,help="Travel preferences is required",location='json')
        req.add_argument('Budget',type=str,help="Budget is required",location='json')
        req.add_argument('County',type=str,help="County is required",location='json')



        req.add_argument('age',type=int,help="Age is required",location='json')
        req.add_argument('gender',type=str,help="Gender is required",location='json',choices=['male','female','other'])
        req.add_argument('married',type=str,help="Married is required",location='json',choices=['yes','no'])
        req.add_argument('graduated',type=str,help="Graduated is required",location='json',choices=['yes','no'])
        req.add_argument('job',type=str,help="Job is required",location='json')
        req.add_argument('family_size',type=int,help="Family size is required",location='json')
        req.add_argument('spending',type=str,help="Spending is required",location='json',choices=['low','average','high'])


        

        
        # add the preferences to the user from the server to the parser
        # Climate_preference: '',
        # Preferred_cuisines: '',
        # Specific_food_categories: '',
        # Preferred_activities: '',
        # Travel_Preferences: '',
        # Budget: '',
        # County:'',

        # add personal data to the parser
        # age: null,
        # gender: '',
        # married: '',
        # graduated: '',
        # job: '',
        # family_size: 1,
        # spending:'',


        args=req.parse_args()
        
        query = {"$or": [{"userid": args['userid']}]}
            # print("come to try")
        try:
            user = self.collection.find_one(query, limit=1)

            if not user:
                
                return Response("User does not exist", status=404)
            
            # print(user['preferences'])
            # print(type(user))
            
            

            if(args['p']=='personal_data'):
                # "age": 0,
                # "gender": "",
                # "married": "",
                # "graduated": "",
                # "job": "",
                # "family_size": 1,
                # "spending":
                update_object={
                    "age": args.get('age',user['personal_data']['age']),
                    "gender": args.get('gender',user['personal_data']['gender']),
                    "married": args.get('married',user['personal_data']['married']),
                    "graduated": args.get('graduated',user['personal_data']['graduated']),
                    "job": args.get('job',user['personal_data']['job']),
                    "family_size": args.get('family_size',user['personal_data']['family_size']),
                    "spending": args.get('spending',user['personal_data']['spending']),

                }
                result = self.collection.update_one(query, {"$set": {"personal_data": update_object}})
                return Response("Updated", status=200)
            elif(args['p']=='preferences'):
                update_object={
                    "Climate_preference": args.get('Climate_preference',user['preferences']['Climate_preference']),
                    "Preferred_cuisines": args.get('Preferred_cuisines',user['preferences']['Preferred_cuisines']),
                    "Specific_food_categories": args.get('Specific_food_categories',user['preferences']['Specific_food_categories']),
                    "Preferred_activities": args.get('Preferred_activities',user['preferences']['Preferred_activities']),
                    "Travel_Preferences": args.get('Travel_Preferences',user['preferences']['Travel_Preferences']),
                    "Budget": args.get('Budget',user['preferences']['Budget']),
                    "County": args.get('County',user['preferences']['County']),

                }

                result = self.collection.update_one(query, {"$set": {"preferences": update_object}})
                

                return Response("Updated", status=200)
        
            else:
                return Response("Invalid value for P", status=400)
        
        except Exception as e:
            print(e)
            return Response("Internal Server Error"+str(e), status=500)


class PredictUserController(Resource):

    def __init__(self):
        self.db=current_app.config['DATABASE']
        self.database = self.db.get_database('users')
        self.collection = self.database.get_collection('users')
        self.model=load('models/user_cat.joblib')
        self.firestoreDB=firestore.client()
        self.catagories=[
            'historical','hotels','waterfall','mountain','rivers','beach',
            'cenama','museum','zoo','park','amusement park','aquarium','art gallery',
            'bakery','cafe','bar','restaurant','food court','grocery store','supermarket','shopping mall','clothing store','shoe store','jewelry store','electronics store','furniture store','book store','liquor store','convenience store','home goods store','department store','pharmacy','hardware store','pet store','florist','toy store','cosmetics store','sporting goods store','thrift store','antique store','mobile phone store','home improvement store','beauty supply store','dollar store','bicycle store','computer store','music store','pawn shop','pop-up shop'
        ]

    def get(self):
        
        parser=reqparse.RequestParser()
        parser.add_argument('uid',type=str,required=True,help="User ID is required",location='args')
        parser.add_argument('email',type=str,required=True,help="Email is required",location='args')

        args=parser.parse_args()
        print(args)

        uid=args['uid']
        email=args['email']

        query = {"$or": [{"email": email}, {"userid": uid}]}
        # print("come to try")
        user = self.collection.find_one(query, limit=1)

        if not user:
            return Response("user not found! ",status=404)
        
        resp_user={
            'preferences':user['preferences'],
            'personal_data':user['personal_data'],
        }
        return jsonify(resp_user)

    def post(self):
        print('came to post')
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('uid', type=str, help='uid can not be empty',required=True,location='json')

        KEY=os.getenv('GOOGLE_API_KEY')

        parser.add_argument('email', type=str, help='email can not be empty',required=True,location='json')
        parser.add_argument('lat', type=float, help='latitude is required',required=True,location='json')
        parser.add_argument('lng', type=float, help='longitude is required',required=True,location='json')
        args = parser.parse_args()

        uid=args['uid']
        email=args['email']
        
        query = {"$or": [{"email": email}, {"userid": uid}]}
        # print("come to try")
        user = self.collection.find_one(query, limit=1)

        if not user:
            return Response("user not found! ",status=404)
        
        resp_user={
            'userid':user['userid'],
            'email':user['email'],
            'preferences':user['preferences'],
            'personal_data':user['personal_data'],
        }
        
        for key in resp_user['preferences']:
            if resp_user['preferences'][key]=='' or resp_user['preferences'][key]==None:
                return Response("Please update your preferences",status=200)
        
        for key in resp_user['personal_data']:
            if resp_user['personal_data'][key]=='' or resp_user['personal_data'][key]==None:
                return Response("Please update your personal data",status=200)
        


        personal_data_df = pd.DataFrame(resp_user['personal_data'],index=range(len(resp_user['personal_data'])))
        preferences_df = pd.DataFrame(resp_user['preferences'],index=range(len(resp_user['preferences'])))

        # Combine the DataFrames
        combined_df = pd.concat([personal_data_df, preferences_df], axis=1)

        # print(combined_df)
        new_df=combined_df.drop(columns=['age','job','County'],axis=1)
        # print(new_df)
        
        for column in new_df.columns:
            # print(column)
            le = LabelEncoder()
            # print(column)

            new_df[column] = le.fit_transform(new_df[column])
        
        # print(combined_df)
        scaler=StandardScaler()
        scaled_data=scaler.fit_transform(new_df)
        row=scaled_data[0]

        predict=self.model.predict(np.array([row]).reshape(1,-1))[0]

        categories_for_user_type={
            'A':random.sample(self.catagories,10),
            'B':random.sample(self.catagories,10),
            'C':random.sample(self.catagories,10),
            'D':random.sample(self.catagories,10),
            
        }

        # stores=list(map(lambda x:x.to_dict(),self.firestoreDB.collection('stores').stream()))

        txt=','.join(categories_for_user_type[predict.upper()])
        print(txt)
        url='https://maps.googleapis.com/maps/api/place/nearbysearch/json'

        google_resp=requests.get(url,
            params={
                'keyword':txt,
                'location':f'{args['lat']},{args['lng']}',
                'radius':50000,
                'key':KEY
            }     
        )
        res=google_resp.json()

        print('map results =',type(res))
        print('predicated categories =',categories_for_user_type[predict.upper()])
        resp_user['stores']=res.get('results',[])

        print('final response =',resp_user)
        
        resp=make_response((jsonify(resp_user),201))

        return resp

class ImageProcesingController(Resource):

    def __init__(self) -> None:
        super().__init__()
        self.firestoreDB=firestore.client()
        


    def get(self):
        return jsonify({"msg": "hello world"})
    
    def post(self):

        model=load('models/fruit_vegetable_model.joblib')
        # print('came')
        # print(request.files)

        if 'file' not in request.files:
            print('No file part')
            return Response("No file part", status=404)
        
        if request.files['file']=='':
            print('No file selected for uploading')
            return Response("No file selected for uploading", status=401)
        
        f=request.files['file']
        filepath='uploads/'+f.filename
        f.save(filepath)
        try:
            img=Image.open(filepath)
            img=img.resize((224,224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            img = tf.expand_dims(img, axis=0)  # Add batch dimension

            predictions = model.predict(img)
            # print(predictions)
            best_one=np.argmax(predictions[0])
            # print(best_one)
            Labels= ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

            print('label count',len(Labels))
            prediction=Labels[best_one]
            print('prediction',prediction)
            
            # get the stoes from the firebase 
            stores=list(map(lambda x:x.to_dict(),self.firestoreDB.collection('stores').stream()))
            resp={}
            resp['prediction']=prediction
            resp_shops=[]
            # print('came eher')
            cats={
                'vegitables':['carrot','beetroot','turnip','sweetcorn','corn','cabbage','soy beans','cucumber','onion','lettuce','garlic','bell pepper','paprika','potato','capsicum','tomato','spinach','raddish','ginger'],
                'fruits':['orange','banana','peas','eggplant','pineapple','pear','grapes','apple','pomegranate','watermelon','lemon','sweetpotato','kiwi','mango','chilli pepper']
            }
            predicted_category=''
            for key in cats.keys():
                if prediction.lower() in cats[key]:
                    predicted_category=key
                    resp['category']=predicted_category
                    break

            for shop in stores:
                # print(shop)
                if shop['selectedCategory'].lower().find(predicted_category.lower())!=-1 or shop['selectedCategory'].lower()==predicted_category.lower():
                    resp_shops.append(shop)
                
                if(shop.get('items')==None):
                    continue
                    
                items=shop['items']
                
                
                for item in items:
                    if item['category'].lower() == predicted_category.lower() or item['category'].lower().find(predicted_category.lower())!=-1:
                        resp_shops.append(shop)
                    if item['name'].lower().find(prediction.lower())!=-1:
                        resp_shops.append(shop)
                    
            
            resp['stores']=resp_shops
            
            print('final response',resp)

            response=make_response(jsonify(resp),201)

            return response

            
        except FileNotFoundError:
            print('file not found')
            return Response("File not found", status=404)
        except Exception as e:
            print(e)
            return Response("Internal Server Error", status=500)

        
class GetRecomendedTopicsController(Resource):

    def __init__(self) -> None:
        super().__init__()

        self.chat:OpenAI=current_app.config['CHAT']

    def get(self):
        
        parser=reqparse.RequestParser()
        parser.add_argument('lat',type=float,required=True,help="User lat is required",location='args')
        parser.add_argument('lng',type=float,required=True,help="User lng is required",location='args')

        args=parser.parse_args()

        lat=args['lat']
        lng=args['lng']

        try:
            resp=self.chat.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "give comma seperated values for toursit attraction places descriptive title acroding to my latitude and longitude only list"},
                    {"role": "user", "content": f"my lat {lat},my lng {lng}"},
                    
                ]
            )
            print(resp.choices[0].message.content)

            
            return Response(resp.choices[0].message.content,status=200)
        except Exception as e:
            print(e)
            return Response("Internal Server Error", status=500)
