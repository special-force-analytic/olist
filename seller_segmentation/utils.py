from scipy import stats
import numpy as np

def boxcox_transform(df, skew_thd=1, lmbda_dict=dict(), except_cols=None):
    '''
    Function for performing boxcox transform on DataFrame object with given threshold of feature's
    skewness.
    - Unless except_cols parameter is specified, all features will be considered for the
    transformation.
    - Unless lmbda_dict parameter is specified, suitable lamba for boxcox transform will be used.
    '''
    df_copy=df.copy()
    lambda_dict=lmbda_dict.copy()
    if len(lambda_dict)==0:
        except_cols = cast_none_or_str_to_list(except_cols)
        for col in df_copy.columns:
            if (df_copy[col].skew() > skew_thd) and (col not in except_cols):
                df_copy[col], lamb = stats.boxcox(df_copy[col].values+np.ones_like(df[col]))
                lambda_dict[col] = lamb
    else:
        for col,lamb in lambda_dict.items():
            df_copy[col] = stats.boxcox(df_copy[col].values+np.ones_like(df_copy[col]),lmbda=lamb)
            
    return df_copy, lambda_dict

def cast_none_or_str_to_list(var):
    '''
    This function cast None or str to list.
    '''
    if var==None:
        var=[]
    else:
        if type(var) != list:
            var = [var]
    return var