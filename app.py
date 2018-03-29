import sys
from flask import Flask
from flask import jsonify
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse
import pandas as pd
from sklearn.externals import joblib


app = Flask(__name__)
api = Api(app)

string_cols = ['sample_uuid']
int_cols = ['channels']

parser = reqparse.RequestParser()
for col in string_cols:
    parser.add_argument(col, type=str)
for col in int_cols:
    parser.add_argument(col, type=int)


class Predict(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.load_model()

    def get(self):
        arguments = parser.parse_args()
        
        #print('****************** {}'.format(arguments))
       
        
        model_input = pd.DataFrame(arguments, index=[0])
       
        #print('%%%%%%%%%%%%%%%%%{}'.format(model_input))
    
        prob = self.model.predict_proba(model_input[['channels']])[0][0]
        
        label = self.model.predict(model_input[['channels']])[0]
        
        result = {
            'sample_uuid': arguments['sample_uuid'],
            'probability': prob,
            'label': label
        }
        return jsonify(**result)

    def load_model(self):
        return joblib.load('Model1.pkl')


api.add_resource(Predict, '/api/v1/predict')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        host = sys.argv[1]
        port = int(sys.argv[2])
    else:
        host = None
        port = None

    app.run(host=host, port=port, debug=False)
