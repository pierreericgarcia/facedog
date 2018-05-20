# LIBRARIES IMPORT
from flask import Flask
from flask_restful import Api
from flask_cors import CORS

# INTERNAL RESOURCES IMPORT
from resources.predict import Prediction

# APP CONFIGURATION
app = Flask(__name__)
CORS(app)
api = Api(app)

# API ROUTES CREATION
api.add_resource(Prediction, '/predict')

# APP INITIALIZATION AND RUNNING
if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')
