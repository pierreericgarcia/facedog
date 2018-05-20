from flask_restful import Resource, reqparse
from utils.dog_app import dog_app
import werkzeug
from PIL import Image
import io
import requests


class Prediction(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument(
        'url',
        type=str,
        required=True,
        help="Need an image url to analyze."
    )

    def post(self):
        data = Prediction.parser.parse_args()
        url = data['url']
        img_data = requests.get(url).content
        print(img_data)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        prediction = dog_app(image)
        return prediction

    def get(self):
        return { 'success': 'Endpoint working' }