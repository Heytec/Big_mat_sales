#Imports
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import joblib

import pickle
import pandas_profiling
from IPython.display import display, HTML



#The st.cache decorator indicates that Streamlit will perform internal magic so that the data will be downloaded only once and cached for future use.
def get_data():
    data = st.file_uploader("upload datset", type=['csv', 'txt'])
    if data is not None:
        df = pd.read_csv(data)

    return df



#Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df




st.title("BIG MART SALES FORCASTING")
st.markdown("Predicting the Item_Outlet_Sales")




def main():
    menu = ['Analysis', 'Machine learning','Prediction']
    selection = st.sidebar.selectbox("Select", menu)


    if selection=='Analysis':
        st.header("Data Analysis")
        st.markdown("Get Train  Dataset :https://www.kaggle.com/ajaygorkar/big-mart-sales-forcasting")
        #upload data

        data = st.file_uploader("upload datset", type=['csv', 'txt'])
        if data is not None:

            df = pd.read_csv(data)
            #view data
            st.markdown("View the first 5 rows ")
            st.dataframe(df.head())
            # data description
            st.markdown("Descriptive statistics ")
            st.write(df.describe())

            profile = df.profile_report(title='Pandas Profiling Report')
            savedprof=profile.to_file(output_file="fifa_pandas_profiling.html")
            panda_html = pd.read_html('fifa_pandas_profiling.html')
            df2=pd.DataFrame(panda_html)
            #st.write(profile, unsafe_allow_html = True)
            st.markdown("Pandas Profilling")
            st.write(df2, unsafe_allow_html = True)



            #Graph ITEM OUTLET SALES
            f = px.histogram(y=df["Item_Outlet_Sales"],x=df["Outlet_Identifier"], nbins=15, title="Item_Outlet_Sales per outlet")
            f.update_xaxes(title="Outlet_Identifier")
            f.update_yaxes(title="Item_Outlet_Sales")
            st.plotly_chart(f)

            # Graph ITEM OUTLET SALES
            f = px.histogram(x=df.Item_Outlet_Sales,title="Item_Outlet_Sales ")
            f.update_xaxes(title="Item_Outlet_Sales")
            f.update_yaxes(title="frequency")
            st.plotly_chart(f)

            # Graph establishment sales
            f = px.histogram(y=df["Item_Outlet_Sales"],x=df["Outlet_Establishment_Year"], nbins=15,
                             title="Item_Outlet_Sales per outlet")
            f.update_xaxes(title="Establishment_Year")
            f.update_yaxes(title="Item_Outlet_Sales")
            st.plotly_chart(f)






    if selection=='Machine learning':
        st.header("Machine Learning")
        st.markdown("Predict sales")
        data = st.file_uploader("upload datset", type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            #st.write(df.info(verbose=True))
            st.markdown("Check missing value ")
            st.write(df.isnull().sum())
            st.markdown("Data Types")
            st.write(df.dtypes)

            st.markdown("Columns to drop ")
            cols = ["Item_Identifier", "Outlet_Size",]
            st_ms = st.multiselect("Columns", df.columns.tolist(),default=cols)
            df1 = df.drop(st_ms, axis=1)
            st.markdown("Original dataset shape before dropping columns ")
            st.write(df.shape)
            st.markdown("Dataset shape after dropping columns")
            st.write(df1.shape)
            dataset_encoded = dummyEncode(df1)

            st.markdown("Check dtypes after converting  to numeric  ")
            st.write(dataset_encoded.dtypes)
            st.markdown("Check missing value")
            st.write(dataset_encoded.isnull().sum())
            dataset_encoded.dropna(subset=['Item_Weight'], inplace=True)
            st.markdown("Check after removing missing value")
            st.write(dataset_encoded.isnull().sum())


            # Split data into training and validation
            df_val = dataset_encoded[dataset_encoded["Outlet_Establishment_Year"] == 2009]
            df_train = dataset_encoded[dataset_encoded["Outlet_Establishment_Year"] != 2009]
            st.markdown("Split data into training and validation ")
            st.write(len(df_val), len(df_train))

            X_train, y_train = df_train.drop("Item_Outlet_Sales", axis=1), df_train.Item_Outlet_Sales
            X_valid, y_valid = df_val.drop("Item_Outlet_Sales", axis=1), df_val.Item_Outlet_Sales

            st.markdown("(X_train.shape),( y_train.shape), (X_valid.shape), (y_valid.shape) ")
            st.write(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

            # Change max_samples value
            model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              min_samples_leaf=300)
            # Cutting down on the max number of samples each estimator can see improves training time
            model.fit(X_train, y_train)

            preds = model.predict(X_valid)
            st.markdown("Mean_absolute_error ")
            st.write(mean_absolute_error(y_valid, preds))
            st.markdown("R2 score ")
            st.write(r2_score(y_valid, preds))

            # save the model to disk
            filename = 'finalized_model.sav'
            joblib.dump(model, filename)
























    if selection=='Prediction':
        st.header("Data Analysis")
        st.markdown("Get Test Dataset :https://www.kaggle.com/ajaygorkar/big-mart-sales-forcasting")

        st.header("Predictions")
        st.markdown("Sales Forecast")
        data1 = st.file_uploader("upload datset", type=['csv', 'txt'])
        if data1 is not None:
            newcustomer1= pd.read_csv(data1)
            st.dataframe(newcustomer1.head())
            newcustomer1.dropna(subset=['Item_Weight'], inplace=True)
            newcustomer2 = newcustomer1.drop(['Item_Identifier', 'Outlet_Size'], axis=1)
            dataset_encoded12 = dummyEncode(newcustomer2)
            #preds = model.predict(dataset_encoded12)

            filename = 'finalized_model.sav'
            loaded_model = joblib.load(filename)
            result = loaded_model.predict(dataset_encoded12)

            predictions = pd.DataFrame( result)
            st.markdown("Predictions")
            st.write(predictions)
            st.markdown("Prediction Forecast")

            st.line_chart(predictions)

















if __name__ == '__main__':
    main()

