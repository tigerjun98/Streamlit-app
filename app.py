# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
import pickle
import xgboost as xgb
import dill
import base64
import dill
import joblib
dill.dumps('foo')

# ML Packages
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import Booster
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from lightgbm.sklearn import LGBMClassifier


def main():

	# @st.cache(show_spinner=False)

	"""Semi Automated ML App with Streamlit """

	activities = ["Prediction outcome","EDA","Plots","Model Building","About"]	
	choice = st.sidebar.selectbox("Select Activities",activities)



	if choice == 'EDA':

		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlation Plot(Matplotlib)"):
				plt.matshow(df.corr())
				st.pyplot()

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()


			if st.checkbox("Pie Plot"):
				all_columns = df.columns.to_list()
				column_to_plot = st.selectbox("Select 1 Column",all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()

	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				st.pyplot()
		
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()


	elif choice == 'Model Building':
		st.subheader("Building ML Models")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			# Model Building
			X = df.iloc[:,0:-1] 
			Y = df.iloc[:,-1]
			seed = 7
			# prepare models
			models = []
			models.append(('LR', LogisticRegression()))
			models.append(('LDA', LinearDiscriminantAnalysis()))
			models.append(('KNN', KNeighborsClassifier()))
			models.append(('CART', DecisionTreeClassifier()))
			models.append(('NB', GaussianNB()))
			models.append(('SVM', SVC()))
			# evaluate each model in turn
			
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())
				
				accuracy_results = {"model name":name,"model_accuracy":cv_results.mean(),"standard deviation":cv_results.std()}
				all_models.append(accuracy_results)


			if st.checkbox("Metrics As Table"):
				st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Algo","Mean of Accuracy","Std"]))

			if st.checkbox("Metrics As JSON"):
				st.json(all_models)
			elif choice == 'About':
				st.subheader("About")

	elif choice == 'Prediction outcome':

		Employment_Type_Self_Employed = 0
		Employment_Type_employee = 0
		Employment_Type_employer = 0
		Employment_Type_government = 0
		Credit_Card_types_normal = 0
		Credit_Card_types_platinum = 0
		Property_Type_condominium = 0
		Property_Type_flat = 0
		Property_Type_terrace = 0
		State_Kedah = 0
		State_Penang = 0
		State_Sabah = 0
		State_Sarawak = 0
		State_Selangor = 0
		State_Sembilan = 0
		State_Trengganu = 0
		Decision = 0
		Score = 0


		Credit_Card_Exceed_Months = st.sidebar.number_input('Credit_Card_Exceed_Months', value=1)

		Employment_Type = st.sidebar.selectbox('Employment Type',('Self Employed', 'Employee', 'Employer', 'Government'))

		Loan_Amount = st.sidebar.number_input('Loan_Amount', value=600000)
		Loan_Tenure_Year = st.sidebar.number_input('Loan_Tenure_Year', value=20)
		More_Than_One_Products = st.sidebar.radio( "More Than One Products", ('Yes', 'No'))
	
		Credit_Card_types = st.sidebar.selectbox('Credit Card Types', ('Normal', 'Gold', 'Platinum'))

		Number_of_Dependents = st.sidebar.number_input('Number_of_Dependents', value=2)
		Credit_Card_Exceed_Months = st.sidebar.number_input('Credit_Card_Exceed_Months', value=2)
		Years_to_Financial_Freedom = st.sidebar.number_input('Years_to_Financial_Freedom', value=5)
		Number_of_Credit_Card_Facility = st.sidebar.number_input('Number_of_Credit_Card_Facility', value=2)
		Number_of_Properties = st.sidebar.number_input('Number_of_Properties', value=2)
		Number_of_Bank_Products = st.sidebar.number_input('Number_of_Bank_Products', value=1)
		Number_of_Loan_to_Approve = st.sidebar.number_input('Number_of_Loan_to_Approve', value=1)
	
		Property_Type = st.sidebar.selectbox('Property Types', ('Condominium', 'Flat', 'Terrace'))
		Years_for_Property_to_Completion = st.sidebar.number_input('Years_for_Property_to_Completion', value=10)
		State = st.sidebar.selectbox('State',('Selangor','Johor','Kedah', 'Penang', 'Sabah', 'Sarawak','Sembilan','Trengganu'))
		Number_of_Side_Income = st.sidebar.number_input('Number_of_Side_Income', value=2)
		Monthly_Salary = st.sidebar.number_input('Monthly_Salary', value=12000)
		Total_Sum_of_Loan = st.sidebar.number_input('Total_Sum_of_Loan', value=800000)
		Total_Income_for_Join_Application = st.sidebar.number_input('Total_Income_for_Join_Application', value=17000)


		col = ['Credit_Card_Exceed_Months','Employment_Type','Loan_Amount','Loan_Tenure_Year',
       'More_Than_One_Products','Credit_Card_types','Number_of_Dependents',
        'Years_to_Financial_Freedom','Number_of_Credit_Card_Facility','Number_of_Properties',
       'Number_of_Bank_Products','Number_of_Loan_to_Approve','Property_Type',
       'Years_for_Property_to_Completion','State','Number_of_Side_Income',
       'Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application',
       'Decision','Score']

		data = [Credit_Card_Exceed_Months,Employment_Type,Loan_Amount,Loan_Tenure_Year,
		More_Than_One_Products,Credit_Card_types,Number_of_Dependents,
		Years_to_Financial_Freedom,Number_of_Credit_Card_Facility,Number_of_Properties,
		Number_of_Bank_Products,Number_of_Loan_to_Approve,Property_Type,
		Years_for_Property_to_Completion,State,Number_of_Side_Income,
		Monthly_Salary,Total_Sum_of_Loan,Total_Income_for_Join_Application,
		Decision,Score]

		numpy_data = np.array([data,data])
		df = pd.DataFrame(data=numpy_data, columns=col)
		df['More_Than_One_Products'] = df['More_Than_One_Products'].astype('category')
		df['Employment_Type'] = df['Employment_Type'].astype('category')
		df['Credit_Card_types'] = df['Credit_Card_types'].astype('category')
		df['Property_Type'] = df['Property_Type'].astype('category')
		df['State'] = df['State'].astype('category')
		df['Employment_Type'] = df['Employment_Type'].cat.codes
		df['Credit_Card_types'] = df['Credit_Card_types'].cat.codes
		df['Property_Type'] = df['Property_Type'].cat.codes
		df['State'] = df['State'].cat.codes
		df['More_Than_One_Products'] = df['More_Than_One_Products'].cat.codes

		convert_dict = {'Credit_Card_Exceed_Months':int,'Employment_Type':int,'Loan_Amount':float,'Loan_Tenure_Year':int,
		'Credit_Card_types':int,'Number_of_Dependents':int,
		'Years_to_Financial_Freedom':int,'Number_of_Credit_Card_Facility':int,'Number_of_Properties':int,
		'Number_of_Bank_Products':int,'Number_of_Loan_to_Approve':float,'Property_Type':int,
		'Years_for_Property_to_Completion':int,'State':int,'Number_of_Side_Income':float,
		'Monthly_Salary':float,'Total_Sum_of_Loan':float,'Total_Income_for_Join_Application':float,
		'Decision':int,'Score':int}

		st.subheader("Prediction outcome： Are you eligible for a loan?")
		model = joblib.load('lgm_model.pkl')

		df = df.astype(convert_dict) 
		train_df = pd.get_dummies(df, drop_first=True)
		X = train_df.drop(columns='Decision')
		y = train_df['Decision']
		X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state =42)
		lgbm_model = LGBMClassifier()
		lgbm_model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		result = y_pred[0]

		if result == 1:
			st.write('You are **approve**. to get the loan total', Loan_Amount)
	
		else:
			st.write('You are **rejected**. to get the loan total', Loan_Amount)
	
	
pipreqs C:\Users\theti\Desktop\data_mining

if __name__ == '__main__':
	main()