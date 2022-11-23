from flask import Flask, render_template,request
import numpy as np
import pickle

model= pickle.load(open('ML_model.pkl','rb'))
app =Flask(__name__)

@app.route('/')
def web():
    return render_template('web.html')

@app.route('/Prediction',methods=['POST']) # this line will be activated when Submit button has been
def Prediction():
    
    input=  request.form.get(" fname" )    
    x=np.array(input)
    Weight=model.predict(x.reshape(1,-1))    

    return render_template('web.html',result='Your predicted weight is {} in KG'.format(Weight))


if __name__ == "__main__":
    app.run(debug=True)    
