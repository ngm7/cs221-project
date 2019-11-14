"""This is a Flask App which starts the App and hosts on a port using which one can enable a Webservice for ease of input.
It starts its own flask server using which we can post into the service
"""
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#This endpoint is just to test if the App is up or not
@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello There !</p>"

#This endpoint is exposed to get a file and pass it to the CLasssifier which if classified correctly should call the Impact Score analyzer.
@app.route('/classifyandroute', methods=['POST'] )
def classifyandscore(self, request):
    return "<h1>To be implemented !</h1>"


app.run()