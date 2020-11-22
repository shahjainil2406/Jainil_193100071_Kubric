import requests
import pandas
import scipy
import numpy
import sys
from scipy import stats

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    url_content = response.content
    csv_file = open('train.csv', 'wb')
    csv_file.write(url_content)
    csv_file.close()
    df = pandas.read_csv('train.csv', header = None).T
    h = df.iloc[0]
    df.drop([0], axis = 0, inplace = True)
    df.columns = h
    
    # getting training X and y
    X = numpy.asarray(df.area.astype('float'))
    y = numpy.asarray(df.price.astype('float'))

    # getting value of slope and intercept for the regression line 
    slope, intercept, r_val, p_val, std_error = stats.linregress(X, y)
    y_pred = area * slope + intercept
    
    return y_pred 


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
