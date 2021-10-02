from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base.svg')
    else:
        text = request.form['text']
        path = "app/static/{}.svg".format(uuid.uuid4().hex)
        make_picture('app/AgesAndHeights.pkl', load('app/model.joblib'), floats_string_to_np_arr(text), path)
        return render_template('index.html', href=path[4:])


def make_picture(training_data_filename, model, new_inp_np_arr, ouput_file):
  data = pd.read_pickle(training_data_filename)
  data = data[data['Age'] > 0]
  ages = data['Age']
  heights = data['Height']

  x_new = np.array(list(range(19))).reshape(19, 1)
  pred = model.predict(x_new)
  fig = px.scatter(x=ages, y=heights, title='Heights v.s. Ages', labels={'x': 'Age (years)', 'y':'Height'})
  fig.add_trace(go.Scatter(x=x_new.reshape(19), y=pred, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)
  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers',
                           marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
  # fig.show()
  # fig.write_image(ouput_file, width=800)
  fig.write_image(ouput_file, width=800, engine='kaleido')


def floats_string_to_np_arr(float_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False

  floats = [float(x) for x in float_str.split(',') if is_float(x)]
  out = np.array(floats).reshape(len(floats), 1)
  return out
