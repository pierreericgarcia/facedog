from flask_restful import Resource, reqparse
from utils.dog_app import dog_app
import werkzeug
from PIL import Image
import io


class Prediction(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument(
            'picture',
            type=werkzeug.datastructures.FileStorage,
            location='files')
        args = parser.parse_args()
        image_data = args['picture'].read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        prediction = dog_app(image)
        return prediction

    def get(self):
        return { 'success': 'Endpoint working' }