import pandas as pd


def load_dataset(dataset, sufix, dif_columns):
    ouput_name = '{}.csv'.format(dataset[:-4])
    df_train = pd.read_csv(ouput_name + sufix + '-train')
    df_test = pd.read_csv(ouput_name + sufix + '-test')

    df2_train = df_train[df_train.columns.difference(dif_columns)]
    df2_test = df_test[df_test.columns.difference(dif_columns)]
    return df_train, df_test, df2_train, df2_test
