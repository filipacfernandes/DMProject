import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def mean_method(df_in, col_name):

    # Choose the data that we are going to use:
    my_data = df_in[['Customer Identity', col_name]].apply(pd.to_numeric)
    
    #-------------------------------------------------------------------------------------------------------------    
    #If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
    imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    
    #fit_transform(self, X[, y]----> Fit to data, then transform it.
    #ravel()-----> A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
    
    my_data[col_name] = imp_mean.fit_transform(my_data[[col_name]]).ravel()
    my_data[col_name] = my_data[col_name].astype(int)
    
    df_in.drop([col_name], axis = 1, inplace = True)
    df_in.insert(1,col_name, my_data[col_name], True)

    return df_in


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------


def frq_method(df_in, col_name):

    # Choose the data that we are going to use:
    my_data = df_in[['Customer Identity', col_name]].apply(pd.to_numeric)
    
    #-------------------------------------------------------------------------------------------------------------    
    #If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    #fit_transform(self, X[, y]----> Fit to data, then transform it.
    #ravel()-----> A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
    
    my_data[col_name] = imp_mean.fit_transform(my_data[[col_name]]).ravel()
    my_data[col_name] = my_data[col_name].astype(int)
    
    df_in.drop([col_name], axis = 1, inplace = True)
    df_in.insert(1,col_name, my_data[col_name], True)

    return df_in

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def pred_model(df_in, col_name):
    
    #‘distance’ : weight points by the inverse of their distance. in this case,
    # closer neighbors of a query point will have a greater influence than neighbors which are further away.
    clf = KNeighborsClassifier(3,weights = 'distance', metric = 'euclidean')
    
    #-----------------------------
    #Separate in two variables the people with a dependent value and without it
    my_data = df_in[['First Policy\'s Year',col_name,'Gross Monthly Salary','Geographic Living Area']]
    
    
    my_data_incomplete = my_data.loc[my_data[col_name].isnull()]
    my_data_complete = my_data[~my_data.index.isin(my_data_incomplete.index)]
    my_data_complete[col_name] = my_data_complete[col_name].astype('str')
    
    #-----------------------------
    #fit(self, X, y) ----> Fit the model using X as training data and y as target values
    #This model will train with the values of 'age' and 'frq' and the values of dependents of
    #the people with the data complete.
    trained_model = clf.fit(my_data_complete.loc[:,['First Policy\'s Year','Gross Monthly Salary']], my_data_complete.loc[:,col_name])
    
    #Predict the values of the data incomplete
    imputed_values = trained_model.predict(my_data_incomplete.drop(columns = ['Geographic Living Area',col_name]))
    
    #Create a data frame with those values.
    temp_df = pd.DataFrame(imputed_values.reshape(-1,1), columns = [col_name])
    
    # Add the new values to the dataframe
    j = 0
    for i in df_in[df_in[col_name].isnull()].index :
        df_in[col_name][i] = temp_df[col_name][j]
        j += 1
        
    return df_in


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
    