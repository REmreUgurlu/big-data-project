import pandas as pd

class DataAccess:
    
    # name is either "ratings" or "animes"
    def get(name):
        df = pd.read_csv(f"{name}.csv")
        return df

    # use this function to get only a portion of the data
    def get_portion(name, nrows):
        df = pd.read_csv(f"{name}.csv", nrows=nrows)
        return df
    
    # this function is used to split the dataset into train, test and validation sets.
    # def split_dataset(name,train_rate=0.8, validate_rate=0.5):
    #     df = DataAccess.get(name)
    #     train_set = df.sample(frac=train_rate, random_state=42)
    #     new_dataset = df.drop(train_set.index)
    #     validation_set =  new_dataset.sample(frac=validate_rate, random_state=42)
    #     test_set = new_dataset.drop(validation_set.index)

    #     return train_set, validation_set, test_set

    # # use this for getting labels and features for your predictions
    # def split_for_prediction(dataset_name, column):
    #     train_set, validation_set, test_set = DataAccess.split_dataset(dataset_name)

    #     train_features = train_set.copy()
    #     validation_features = validation_set.copy()
    #     test_features = test_set.copy()

    #     train_labels = train_features.pop(column)
    #     validation_labels = validation_features.pop(column)
    #     test_labels = test_features.pop(column)

    #     return train_features, train_labels, validation_features, validation_labels, test_features, test_labels

    # def get_ratings():
    #     df = pd.read_csv("ratings.csv")
    #     return df
    
    # def get_animes():
    #     df = pd.read_csv("animes.csv")
    #     return df



        

if __name__ == "__main__":
    # This function reads the "animes.csv" file and loads in the first "9" records. 
    get_df = DataAccess.get_portion("animes", 9)
    print(get_df)

    read_df = DataAccess.get("animes")
    print(read_df)