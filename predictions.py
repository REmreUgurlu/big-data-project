from data_access import DataAccess as d_a 
import tensorflow as tf
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

class Predictions:
    
    def preprocessing(data):

        # numerical_features = data.select_dtypes(include=['int64']).columns
        # numerical_features = numerical_features.difference(['anime_id'])
        # data = data.drop(['TV', 'Special', 'Movie', 'OVA'], axis=1)
        X = data.drop(['anime_id','name','rating'], axis=1)
        y= data['rating']

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)

        names_train = data.loc[X_train.index, 'name'].values
        names_test = data.loc[X_test.index, 'name'].values

        return X_train, X_test, y_train, y_test, names_train, names_test
    
    def nn_model(input_shape):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer=keras.optimizers.Adamax(), loss='mean_squared_error')
        return model

    def train_model(model, X_train, y_train, epochs=50, batch_size=32):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate_model(model, X_test, y_test):
        loss = model.evaluate(X_test, y_test)
        print(f'Mean Squarred Error: {loss}')

    def predict(model, X):
        predictions = model.predict(X)
        return predictions


if __name__ == '__main__':
    data = d_a.get('animes')
    # data = data.drop(['TV', 'Special', 'Movie', 'OVA'], axis=1)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train, X_test, y_train, y_test, names_train, names_test = Predictions.preprocessing(data)

    input_shape = X_train.shape[1]
    model = Predictions.nn_model(input_shape=input_shape)
    training = Predictions.train_model(model, X_train, y_train)
    evaluation = Predictions.evaluate_model(model, X_test,y_test)

    # Select a subset of data for prediction (for example, first 10 rows)
    data_to_predict = data.iloc[:10].copy()  # Adjust as needed

    # Extract 'name' column for later use
    names_to_predict = data_to_predict['name'].values

    actual_ratings = data_to_predict['rating'].values

    # Preprocess the data for prediction
    X_to_predict = data_to_predict.drop(['name','anime_id','rating'], axis=1)  # Exclude 'name' from features

    # Make predictions on the selected subset
    predictions = Predictions.predict(model, X_to_predict)

    # Combine names and predictions into a DataFrame
    results_df = pd.DataFrame({'Name': names_to_predict, 'PredictedRating': predictions.flatten(), 'ActualRating': actual_ratings})

    # Print or use the results as needed
    print(results_df)

        