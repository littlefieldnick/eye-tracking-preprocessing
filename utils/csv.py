import pandas as pd

def load_csv(pth, columns_to_include=None, low_memory=False):
    try:
        if columns_to_include:
            return pd.read_csv(pth, usecols=columns_to_include, low_memory=low_memory)

        return pd.read_csv(pth, low_memory=low_memory)
    except Exception as e:
        print(e)

def df_to_csv(data, out_pth):
    try:
        if data is None:
            print("There is no data to save given the provided configuration... Exiting")
            exit(0)

        data.to_csv(out_pth, index=False)
    except Exception as e:
        print(e)

