from logging import warning
from joblib.numpy_pickle import load
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_dict import *
import joblib
import os

st.beta_set_page_config(page_title='Severity Prediction', page_icon = '🚘',  initial_sidebar_state = 'auto')
st.markdown("<div style='background-color:black;'><h3 style='text-align:center; font-size:40px;color:white'><b>Road Collision Severity Prediction</b></h3></div>", unsafe_allow_html=True)
st.header("Seattle Department of Transportation")

menu = ['Home','Plots','Prediction']

choice = st.selectbox("Choose your menu here",menu)

feature_names = ['Address Type','Junction Type','Drunken','Weather','Road Condition','Light Condition','Pedestrain way granted','Speedy Vehicle']


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


if choice == 'Home':
    st.header("Agenda")
    st.write("""The Seattle government is going to prevent avoidable car accidents by employing methods that alert drivers, 
                health system, and police to remind them to be more careful in critical situations.
                In most cases, not paying enough attention during driving, abusing drugs and alcohol or driving at very high speed are 
                the main causes of occurring accidents that can be prevented by enacting harsher regulations. 
                \n Besides the aforementioned reasons, weather, visibility, or road conditions are the major uncontrollable factors that can 
                be prevented by revealing hidden patterns in the data and announcing warning to the local government, police and drivers on 
                the targeted roads.
                \n The target audience of the project is local Seattle government, police, rescue groups, and last but not 
                least, car insurance institutes. The model and its results are going to provide some advice for the target audience to make 
                insightful decisions for reducing the number of accidents and injuries for the city""")

elif choice == 'Plots':
    st.markdown("<div style='background-color:lightblue;'><h3 style='text-align:center; font-size:30px'><b>Data Visualization</b></h3></div>", unsafe_allow_html=True)
    
    df = pd.read_csv("data/data_cleaned.csv", usecols=['ADDRTYPE','SEVERITYCODE','COLLISIONTYPE','JUNCTIONTYPE','UNDERINFL','WEATHER','ROADCOND','LIGHTCOND','SPEEDING','year'])
    # df['SEVERITYCODE'].replace(2,1,inplace=True)
    # df['SEVERITYCODE'].replace(3,2,inplace=True)
    df['SEVERITYCODE'].replace(4,3,inplace=True)
    st.dataframe(df.head(10))

    if st.checkbox("Show severities code bar graph"):
        fig = plt.figure()
        code = ['1','2','3']
        freq = df['SEVERITYCODE'].value_counts()
        plt.bar(code, freq)
        st.pyplot(fig)
        st.write("frequency of severity codes:",df['SEVERITYCODE'].value_counts())
    if st.checkbox("Severity vs Drunken"):
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['SEVERITYCODE'], hue='UNDERINFL', data=df, palette='Set1')        
        st.pyplot(fig)
        st.write("This shows very few were drunken while driving.")
    if st.checkbox("Severity vs Road Condition"):
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['SEVERITYCODE'], hue='ROADCOND', data=df, palette='Set1')        
        st.pyplot(fig)
        st.write("Most accidents happened on Dry and Wet road.")
            
    if st.checkbox("Severity vs Weather"):
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['SEVERITYCODE'], hue='WEATHER', data=df, palette='Set3')   
        plt.legend(loc='upper right', prop={'size':10})     
        st.pyplot(fig)    
        st.write("Most accidents happened on Clear day.")
    if st.checkbox("Severity vs Road light condition"):
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['SEVERITYCODE'], hue='LIGHTCOND', data=df, palette='Set3')   
        plt.legend(loc='upper right', prop={'size':10})     
        st.pyplot(fig)    
        st.write("Most accidents happened daylight and at night when street lights were on.\
                \n Very few accidents happened when light were off at night.")
    if st.checkbox("Severity vs Speed"):
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['SEVERITYCODE'], hue='SPEEDING', data=df, palette='Set1')   
        plt.legend(loc='upper right', prop={'size':10})     
        st.pyplot(fig)    
        st.write("**Few accidents happened at high speed as compared to low speed.**")
    if st.checkbox("Severity vs Year"):    
        fig = plt.figure(figsize=(10,5))
        sns.countplot(x=df['year'], hue='SEVERITYCODE', data=df, palette='Set2')   
        plt.legend(loc='upper right', prop={'size':10})     
        st.pyplot(fig)    
        st.write("**Few accidents happened of severity code 2 every year.**")

elif choice == 'Prediction':
        st.markdown("<div style='background-color:lightblue'><h3 style='text-align:center;  font-size:30px'><b>Predictive analysis</b></h3></div>", unsafe_allow_html=True)

        left,mid,right = st.beta_columns(3)

        with left:
        # <-------------------Inputs---------------------------->
            addrtype = st.radio("What was in front of you!", tuple(address_dict.keys())) 
            col_type = st.radio("What was the Collision type", tuple(collision_type.keys()))
            no_person = st.slider("Person involved in accident",min_value=0, max_value=50)
            no_ped = st.slider("Pedestrians involved in accident",min_value=0, max_value=6)
            no_cycle = st.slider("Cycles involved in accident",min_value=0, max_value=2)
            no_veh = st.slider("Number of vehicles involved in accident",min_value=0, max_value=15)
            junc = st.radio('What kind of junction is in front of you', tuple(junction_dict.keys()))
            attention = st.radio('Collision happened due to inattention', tuple(yes_no.keys()))
            no_injuries = st.slider("Number of Injured people",min_value=0, max_value=100)
   
        with mid:
            pass
            
        with right:
            drunken = st.radio('Driver is drunken or not', tuple(yes_no.keys()))
            weath_cond = st.radio("Weather Condition",tuple(weather_dict.keys()))
            road_cond = st.radio("How's the road condition", tuple(road_condition.keys()))
            light_cond = st.radio("How's the Light condition", tuple(light_condition.keys()))
            ped_grnt = st.radio('Pedestrain right of way was given or not', tuple(yes_no.keys()))
            veh_speed = st.radio('Whether vehicle was in speed or not', tuple(yes_no.keys()))
            veh_park = st.radio('Does moving vehicle hit any parked vehicle', tuple(yes_no.keys()))
            # no_fatalities = st.slider("Number of Fatalities",min_value=0, max_value=50)
            no_serious_injuries = st.slider("Number of serious injured people",min_value=0, max_value=50)
            
        feature_list = [
                        get_value(addrtype,address_dict),
                        get_value(col_type, collision_type),
                        no_person,
                        no_ped,
                        no_cycle,
                        no_veh,
                        no_injuries,
                        no_serious_injuries,
                        get_value(junc, junction_dict),
                        get_value(attention, yes_no),
                        get_value(drunken, yes_no),
                        get_value(weath_cond, weather_dict),
                        get_value(road_cond, road_condition),
                        get_value(light_cond, light_condition),
                        get_value(ped_grnt, yes_no),
                        get_value(veh_speed, yes_no),
                        get_value(veh_park, yes_no)                   
        ]

        single_sample = np.array(feature_list).reshape(1,-1)
        st.write("Sample",single_sample) 
        # st.code(feature_list)

        model_choice = st.selectbox("Select Model",["Logistic Regression","KNN","XGBoost Classifier","Decision Tree Classifier", "Random Forest Classifier"])
        st.write(" ")
        if st.button("Predict"):
            if model_choice == "Logistic Regression":
                loaded_model = load_model('models/log_reg_model.pkl')
                prediction = loaded_model.predict(single_sample)
                prob = loaded_model.predict_proba(single_sample)
                # st.write(model_choice,':',prediction)


            elif model_choice == "KNN":
                loaded_model = load_model('models/knn_model.pkl')
                prediction = loaded_model.predict(single_sample)
                # st.write(model_choice,':',prediction)
                prob = loaded_model.predict_proba(single_sample)


            elif model_choice == "XGBoost Classifier":
                loaded_model = load_model('models/xgb_model.pkl')
                
                prediction = loaded_model.predict(single_sample)
                # st.write(model_choice,':',prediction)

            elif model_choice == "Decision Tree Classifier":
                loaded_model = load_model('models/dtc_model.pkl')
                prediction = loaded_model.predict(single_sample)
                # st.write(model_choice,':',prediction)
                prob = loaded_model.predict_proba(single_sample)

            # elif model_choice == "Random Forest Classifier":
            else: 
                loaded_model = load_model('models/rf_model.pkl')
                prediction = loaded_model.predict(single_sample)
                # st.write(model_choice,':',prediction)
                prob = loaded_model.predict_proba(single_sample)

        
            # if prediction == 4:
            #     st.warning("Fatality")
            if prediction == 3:
                st.error("💀 Serious Injuries")
                st.write("Probability : {}%".format( round((prob[:,2]*100)[0],2) ))
                

            elif prediction == 2:
                st.warning("🤕 Mild Injuries")
                st.write("Probability : {}%".format( round((prob[:,1]*100)[0],2) ))


            elif prediction == 1:
                st.info("💥🚗 Property damage only.")
                # probility = prob[:,0]*100
                st.write("Probability : {}%".format( round((prob[:,0]*100)[0],2) ))

