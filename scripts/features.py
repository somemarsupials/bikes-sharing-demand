import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing

# utility functions


def parse_datetime(dt):
    """Convert time stamp string into datetime object."""
    return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')


def make_dummies(data, column, categories):
    """Given a dataset, a column name and categories, create 
    dummies and drop the original column. Categories should be
    an ordered list, where position indicates encoded value, or
    a dictionary mapping encoding to column name.
    """
    if type(categories) == list:
        pairs = enumerate(categories)
    else:
        pairs = categories.items()
    for index, name in pairs:
        data.loc[:, name] = (data[column] == index)
    del data[column]
    return data


# transformations


def weather(data):
    """Remove rows with weather type 4. There is only one row with this
    weather type.
    """
    return data[data['weather'] != 4]


def datetimes(data):
    """Parse time stamps into datetime objects."""
    data.loc[:, 'datetime'] = data['datetime'].apply(parse_datetime)
    return data


def hours(data):
    """Retrieve hour from datetime objects."""
    data.loc[:, 'hour'] = data['datetime'].apply(lambda dt: dt.hour)
    return data


def years(data):
    """Retrieve year from datetime objects."""
    is_2011 = lambda dt: dt.year == 2011
    data.loc[:, '2011'] = data['datetime'].apply(is_2011)
    return data


def dummy_seasons(data):
    """Convert season to binary dummy variables. We exclude one month
    to avoid the dummy trap.
    """
    seasons = ['winter', 'spring', 'summer']
    return make_dummies(data, 'season', seasons)


def dummy_hours(data):
    """Convert hour to binary dummy variables. We exclude one hour
    to avoid the dummy trap.
    """
    hours = list(map(lambda h: 'hour {}'.format(h), range(23)))
    return make_dummies(data, 'hour', hours)


def dummy_weather(data):
    """Convert weather to binary dummy variables. We exclude one hour
    to avoid the dummy trap.
    """
    types = ['clear', 'cloudy']
    return make_dummies(data, 'weather', types)


# main

BASIC = [
    weather,
    datetimes,
    hours,
    years,
    dummy_seasons,
    dummy_hours,
    dummy_weather,
]


def engineer(data, transformations=BASIC):
    """Apply a series of transformations on a dataset. Return the result
    after the final transformation.
    """
    for transform in transformations:
        data = transform(data)
    return data


if __name__ == '__main__':
    data = engineer(pd.read_csv('./data/train.csv'))
    print(data.columns)
