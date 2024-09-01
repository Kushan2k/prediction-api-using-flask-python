from flask import Flask
from flask_restful import Api
from dotenv import load_dotenv
import os
from resources.SaveUserController import SaveUserController,ImageProcesingController,PredictUserController
from flask_cors import CORS
import connection
import firebase_admin


# configarations
load_dotenv()
app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = os.environ.get("DEBUG")
app.config['DATABASE']=db=connection.DatabaseConnection()
FBapp = firebase_admin.initialize_app(firebase_admin.credentials.Certificate('./foriegn-traveller-firebase-adminsdk-uvg5y-cb35de410f.json'))

# endpoints 
api.add_resource(SaveUserController, '/')
api.add_resource(ImageProcesingController, '/upload')
api.add_resource(PredictUserController, '/predict')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
