import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
data=pd.read_csv("brain_stroke.csv")
data=np.array(data)
x=data[:,:-1]
y=data[:,-1]
y=y.astype('int')
x=x.astype('int')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
lr=LogisticRegression()
lr.fit(x_train,y_train)
imput=[int(x) for x in "4$ 32 68".split(' ')]
final=[np.array(input)]
pickle.dump(lr,open('Sample.pkl','wb'))



#import streamlit as st
import pandas as pd
import numpy as np
import pickle
model=pickle.load(open('Sample.pkl','rb'))
def predict_brain_stroke(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status):
    input=np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:,{1}f}'.format(prediction[0][0],2)
    print(type(pred))
    return float(pred)
def main():
    st.title("Streamlit Tutorial")
    html_temp='''
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Brain Stroke Prediction App </h2>
    </div>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    gender=st.text_input("gender","Type Here")
    age=st.text_input("age","Type Here")
    hypertension=st.text_input("hypertension","Type Here")
    heart_disease=st.text_input("heart_disease","Type Here")
    ever_married=st.text_input("ever_married","Type Here")
    work_type=st.text_input("work_type","Type Here")
    Residence_type=st.text_input("Residence_type","Type Here")
    avg_glucose_level=st.text_input("avg_glucose_level","Type Here")
    bmi=st.text_input("bmi","Type Here")
    smoking_status=st.text_input("smoking_status","Type Here")
    safe_html="""
    <div style="background-color:#F4O03F;padding:10px>
    <h2 style="color:white ;text-align:center;">You does not occur brain stroke</h2>
    </div>
    """
    danger_html="""
    <div style="background-color:#F85050;padding:10px>
    <h2 style="color:black;text-align:center;">You may occur brain stroke</h2>
    </div>
    """
    if st.button("Predict"):
        output=predict_brain_stroke(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
        st.success("The probability of disease taking place is{}".format(output))
        if output>0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
if __name__=='__main__':
    main()

