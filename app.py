# import libraries
import streamlit as st
import pandas as pd
import joblib
# Load the model
model=joblib.load('notebook/iris_model.joblib')


# function to predict species with probablity

def predict_labels(model,sep_length,sep_wid,pet_lrn,pet_wid):
    xnew=[
        {
            "sepal_length":sep_length,
            "sepal_width":sep_wid,
            "petal_length":pet_lrn,
            "petal_width":pet_wid
        }
    ]
    df_xnew=pd.DataFrame(xnew)
    pred=model.predict(df_xnew)
    prob=model.predict_proba(df_xnew)
    res_prob={}
    # get probablity as dict
    for c,p in zip(model.classes_,prob.flatten()):
        res_prob[c]=p.round(4)

    return pred[0],res_prob    


# build the streamlit app
st.set_page_config(page_title="Iris Project")

# add the title to project
st.title("Iris ML Project")

st.subheader("by Rizwan Khan")

#create no inputs
sep_length=st.number_input("Sepal Length :",min_value=0.00,step=0.01) 
sep_wid=st.number_input("Sepal Width :",min_value=0.00,step=0.01) 
pet_lrn=st.number_input("Petal Length :",min_value=0.00,step=0.01) 
pet_wid=st.number_input("Petal Width :",min_value=0.00,step=0.01) 
    


# create button fors streamlit    
button=st.button("Predict",type="primary")

# after clicking button
if button:
    pred,prob=predict_labels(model,sep_length,sep_wid,pet_lrn,pet_wid)
    st.subheader(f'Predicted Species :{pred}')
    for c,p in prob.items():

      st.subheader(f'{c}:{p}')
      st.progress(p)