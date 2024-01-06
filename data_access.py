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