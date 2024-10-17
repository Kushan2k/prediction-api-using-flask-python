from flask import Flask
from flask_restful import Api
from dotenv import load_dotenv
import os
from resources.SaveUserController import SaveUserController,ImageProcesingController,PredictUserController,GetRecomendedTopicsController,GetSuggetionOnPrompt
from flask_cors import CORS
import connection
import firebase_admin
from openai import OpenAI

# configarations
load_dotenv()
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}})

chat=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


app.config['DEBUG'] = os.environ.get("DEBUG")
app.config['DATABASE']=db=connection.DatabaseConnection()
app.config['CHAT']=chat
FBapp = firebase_admin.initialize_app(firebase_admin.credentials.Certificate('./foriegn-traveller-firebase-adminsdk-uvg5y-cb35de410f.json'))

# endpoints 
api.add_resource(SaveUserController, '/')
api.add_resource(ImageProcesingController, '/upload')
api.add_resource(PredictUserController, '/predict')
api.add_resource(GetRecomendedTopicsController,'/recomended')
api.add_resource(GetSuggetionOnPrompt,'/suggetion')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
