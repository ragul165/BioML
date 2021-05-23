from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('MLmodel.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("/forest.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    p=model.predict(final)
    print(int_features)
    print(final)
    print(p)
    print(p[0])
    #output='{0:.{1}f}'.format(p[0], 2)

    if p[0]==2:
        return render_template('forest.html',pred='Very Prone to Corona Diease')
    if p[0]==1:
        return render_template('forest.html',pred='No Prone to Corona Diease')
    if p[0]==0:
        return render_template('forest.html',pred='less Prone to Corona Diease')
    

if __name__ == "__main__":
    app.run(debug=True)