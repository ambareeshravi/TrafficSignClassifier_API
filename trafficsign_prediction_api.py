'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: trafficsign_prediction_api.py
Description:
    Contains the API to classify images by uploading them
'''

from prediction import *

from PIL import Image
import numpy as np
import io

import flask
from flask import Flask

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    '''
    Method to predict an image that is uploaded

    Args:
        -
        
    Returns:
        results as json
    
    Exception:
        -
    '''
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB') 

            # preprocess the image and prepare it for classification
            im = tester.resize_im_array(np.array(image))
            pred = tester.predict_im_array(im)
            data["result"] = pred[-1]
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == '__main__':
    # Define the tester object
    tester = Tester()
    # Run the application
    app.run(host='127.0.0.1', debug=True, port = 12345)
    # To access externally
    # app.run(host='<ip address>', debug=True, threaded=True, use_reloader=False, port = <port>, ssl_context='adhoc')
