from flask import Flask, jsonify, render_template, request
app = Flask(__name__)
@app.route('/')
@app.route('/test')
def my_microservice():
    #return jsonify({'Hello': 'World!'})
    return render_template('index.html')
import numpy as np
from sklearn.externals import joblib
@app.route('/test',methods=['POST','GET'])
def test():
        gbr=joblib.load('model.pkl')
        result=request.form
        squar=result['SQUARE']

        user_input=np.array(squar)
        user_input=user_input.reshape(-1,1)
        user_input=user_input.astype(float)
        #print(user_input)
        #a=input_to_one_hot(user_input)
        price_pred=gbr.predict(user_input)
        price_pred=round(price_pred, 2)
        return render_template('result.html',prediction=price_pred)
#return json.dumps({'price': price_pred})

if __name__ == "__main__":
    app.run(debug=True)

