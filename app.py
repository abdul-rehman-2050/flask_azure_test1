import os
import json
import cv2
import numpy as np
import sklearn


from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/versions', methods=['GET'])
def versions():
    opencv_version = cv2.__version__
    sklearn_version = sklearn.__version__
    numpy_version = np.__version__

    versions_dict = {
        "opencv_version": opencv_version,
        "scikit-learn_version": sklearn_version,
        "numpy_version": numpy_version
    }

    return jsonify(versions_dict)       


if __name__ == '__main__':
   app.run()
