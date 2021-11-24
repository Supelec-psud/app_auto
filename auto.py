import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from PIL import Image


st.write("""
# Car accident risk prediction
by Tae-Heun KIM
""")

img = Image.open('C:/Users/taehe/PycharmProjects/candyapp/images.png')

st.image(img)
st.sidebar.header('User Input Parameters')

def user_input_features():
    INCOME = st.sidebar.slider('Income', 0, 367030, 15000)
    MVR_PTS = st.sidebar.slider('Moving record point', 0, 10, 0)
    URBANICITY = st.sidebar.checkbox('Urbanicity')
    Driving_kids = st.sidebar.checkbox('Driving kids')
    Homeowner = st.sidebar.checkbox('Homeowner')
    Past_claims = st.sidebar.checkbox('Past Claims')
    PARENT1 = st.sidebar.checkbox('Single parent')
    MSTATUS = st.sidebar.checkbox('Already married ?')
    CAR_USE = st.sidebar.checkbox('Commercial car ?')
    TRAVTIME = st.sidebar.checkbox('How much time to work ?')
    REVOKED = st.sidebar.checkbox('Revoked')
    Kids = st.sidebar.checkbox('Is there a child ?')
    Minivan = st.sidebar.checkbox('Is his car a minivan ?')
    stu_blu = st.sidebar.checkbox('Student or Bluecollar worker ?')
    college = st.sidebar.checkbox('More than high school ?')
    Agegroup = st.sidebar.checkbox('Young age')
    CarGroup = st.sidebar.checkbox('New car ?')
    SEX = st.sidebar.checkbox('Male ?')
    data = {'Income':INCOME,
            'Urbanicity': URBANICITY,
            'Driving kids': Driving_kids,
            'Homeowner': Homeowner,
            'Past Claims': Past_claims,
            'Single parent':PARENT1,
            'Already married ?': MSTATUS,
            'Commercial car ? ':CAR_USE,
            'How much time to work ?':TRAVTIME,
            'Revoked':REVOKED,
            'Moving record point':MVR_PTS,
            'Is there a child ?':Kids,
            'Is his car a minivan ?':Minivan,
            'Student or Bluecollar worker ?':stu_blu,
            'More than high school ?':college,
            'Young age':Agegroup,
            'New car ?':CarGroup,
            'Male ?':SEX
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv("C:/Users/taehe/PycharmProjects/candyapp/app_data.csv")


X=data.drop(['JOB','KIDSDRIV','HOMEKIDS','CLM_FREQ','AGE','TARGET_AMT','TARGET_FLAG','YOJ','RED_CAR','TIF','CAR_AGE','CAR_TYPE','EDUCATION','HOME_VAL','INDEX','BLUEBOOK','OLDCLAIM','log_TARGET_AMT','log_INCOME','log_BLUEBOOK','log_OLDCLAIM'],axis=1)
Y=data.TARGET_AMT


clf = RandomForestClassifier(n_estimators=20)
clf.fit(X, Y)

prediction = clf.predict_proba(df)
prediction = (prediction[:,0]*100)

clf2 = RandomForestRegressor(n_estimators=20)
clf2.fit(X, Y)

prediction2 = clf2.predict(df)

def pred(x):
    if 50 <x <70 :
        return " Standard:frowning:"
    elif x<=50 :
        return "Preferred :sparkles:"
    elif x>=70 :
        return "Substandard :frowning:"


st.subheader('Car accident risk')
st.write('Predicted accident probability:', int(prediction), '%$_{\odot}$')
st.subheader('Risk selection')
st.write(pred(prediction))
st.subheader('Compensation')
st.write('Predicted amount of compensation:', int(prediction2), '$')
#st.write(prediction)
#max_depth=110, max_features=3, min_samples_leaf=3,min_samples_split=12

