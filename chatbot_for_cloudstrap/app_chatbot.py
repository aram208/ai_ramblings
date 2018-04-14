from flask import Flask
from flask_restful import Resource, Api, reqparse

import bot_engine

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('question')

cbmodel = bot_engine.TheBot()

class ConversationResource(Resource):
    def post(self):
        args = parser.parse_args()
        return {'response':'{}'.format(cbmodel.get_response(args['question']))}

    def get(self):
        args = parser.parse_args()
        #return {'response': 'You said: {}'.format(args['question'])}
        return {'response':'{}'.format(cbmodel.get_response(args['question']))}

api.add_resource(ConversationResource, '/')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
