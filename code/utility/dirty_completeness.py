import numpy as np

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return  df

def injection(df_pandas, seed, name, name_class):

    np.random.seed(seed)

    #%%

    df_list = []

    perc = [0.50, 0.40, 0.30, 0.20, 0.10]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p,1-p]
        df_dirt = check_datatypes(df_dirt)

        for col in df_dirt.columns:

            if col!=name_class:

                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True,col]=np.nan

        df_list.append(df_dirt)
        print("saved {}-completeness{}%".format(name, round((1-p)*100)))
    return df_list
