from flask_restful import Resource, reqparse
from utils.dog_app import dog_app
import werkzeug
from PIL import Image
import io


class Prediction(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        'url', 
        type=str, 
        required=True, 
        help="I need an image url."
    )

    def post(self):
        data = Prediction.parser.parse_args()
        prediction = dog_app(data["url"])
        return prediction

    def get(self):
        return {'success': 'Endpoint working'}
