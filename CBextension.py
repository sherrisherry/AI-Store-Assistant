import pandas as pd
import pickle, json, os
import datetime
from catboost import CatBoostClassifier, CatBoostRegressor
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fuzzywuzzy import process

data_folder = 'data'
model_folder ='models'
mdb_password = ''

def drop_key(dict, k):
    try:
        del dict[k]
    except KeyError:
        pass
    return dict
    
def docdb_search(query, collection, desc, order=None, top=10):
    uri = f"mongodb+srv://devops:{mdb_password}@meetfresh.kereo5c.mongodb.net/?appName=meetfresh"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client['KOC']
    
    result = None
    try:
        result = db[collection].find(query).sort(order if order else '_id').limit(top)
    except Exception as e:
        print(e)
    result = json.dumps([drop_key(_, '_id') for _ in result], ensure_ascii=False, default = str) if result else f'ERROR: {e}'
    client.close()
    return {
            'value': result,
            'description': desc
            }

def get_date(n):
    date = datetime.datetime.today()+datetime.timedelta(days=n)
    return {
            'value': str(date),
            'description': f"{str(n)}天后"
            }

def get_closest_dish_name(user_input, valid_dish_list, threshold=80):
    match, score = process.extractOne(user_input, valid_dish_list)
    if score >= threshold:
        return match
    else:
        return None

class DishSalesForecaster:
    """
    A forecasting extension to predict dish-level sales using a pre-trained CatBoost model (pickled).
    """

    def sales_prediction(start_date, end_date, temp = 71, dishes = None, promption = 0) -> pd.Series:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S.%f')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S.%f')
        df = pd.read_csv(os.path.join(data_folder,'dishtypes.csv'))
        # Dishes & Customizations Breakdown	Type	item_id	Category	Is_Drink	Is_Dessert	Is_Cold	Is_Signature
        if dishes:
            valid_dishes = df['item_id'].unique().tolist()
            matched_dishes = []
            for dish in dishes:
                match = get_closest_dish_name(dish, valid_dishes)
                if match:
                    matched_dishes.append(match)
            if not matched_dishes:
                return {'value':'0', 'description':'未找到匹配的菜品，无法预测销售额'}
            df = df[df['item_id'].isin(matched_dishes)]
            dishes = '"' + '","'.join(dishes) + '"'
            matched_dishes = '"' + '","'.join(matched_dishes) + '"'
        ds = pd.date_range(start=start_date, end=end_date, freq = 'D').to_series()
        ds.name = 'ds'
        df = df.merge(ds, how = 'cross')
        df['Count'] = 1
        df['Gross Sales'] = 7
        df['Day'] = df['ds'].dt.day_name()
        df['Customization Total'] = 0
        df['temp_feelslike'] = 72
        df['is_holiday'] = 0
        df['YearMonth'] = str(df['ds'].dt.year)+'-'+str(df['ds'].dt.month)
        df['promotions'] = 0
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['week'] = df['ds'].dt.isocalendar().week
        df['lag_1'] = 7
        df['lag_7'] = 0
        df['Is_Cold'] = 1
        cols = ['ds', 'Dishes & Customizations Breakdown', 'Count', 'Gross Sales',
       'Type', 'item_id', 'Category', 'Day', 'Customization Total',
       'temp_feelslike', 'is_holiday', 'YearMonth', 'Is_Drink', 'Is_Dessert',
       'Is_Cold', 'Is_Signature', 'promotions', 'dayofweek', 'month', 'week',
       'lag_1', 'lag_7'] # order matters
        df = df[cols]
        
        with open(os.path.join(model_folder,'best_catboost_model.pkl'), "rb") as f:
            loaded_model = pickle.load(f)

        # input_data = pd.DataFrame([{
        # 'ds':date,
        # 'Dishes & Customizations Breakdown': 'almond pudding qt',
        # 'Count':1,
        # 'Gross Sales':7,
        # 'Type': 'Dish',
        # 'item_id': 'almond pudding qt',
        # 'Category':'qt cup',
        # 'Day': date.weekday(),
        # 'Customization Total':0,
        # 'temp_feelslike':temp,
        # 'is_holiday':0,
        # 'YearMonth':str(date.year)+'-'+str(date.month),
        # 'Is_Drink': 0,
        # 'Is_Dessert':0,
        # 'Is_Cold':0,
        # 'Is_Signature':0,
        # 'promotions':0,
        # 'dayofweek':date.weekday(),
        # 'month':date.month,
        # 'week':date.weekday(),
        # 'lag_1':7,
        # 'lag_7':0
        #     }])

        prediction = loaded_model.predict(df).sum()
        result = {
            'value': str(prediction),
            'description': f"{str(start_date.date())} ~ {str(end_date.date())} {dishes}近似成{matched_dishes}后预测的销售额（美元）" if dishes else f"{str(start_date.date())} ~ {str(end_date.date())} 预测的总销售额（美元）"
            }
        
        return result

