# 업비트 API key(2022.06.30)
ACCESS_KEY = "YtozFZ6R5lgg3gO0ZhxaJpfX5X8avw1PEJShisVF"
SECRET_KEY = "c6A9yadQVob87xnwBXbuVzafxcDiMb3rOv3hozuK"

from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mplfinance as mpf
import telegram
import time
import datetime
import logging
import sys

import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import uuid
import datetime
import time
import json
# import talib as ta
import copy
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.ensemble import RandomForestClassifier

import helper_general

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler(f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
        ]
)

class MyUpbit(helper_general.Exchange):
    def __init__(self, api_key, secret_key) -> None:
        super().__init__(api_key, secret_key)
        logging.info(f"API(or ACCESS) Key : {api_key}")
        logging.info(f"SECRET Key : {secret_key}")

        # 변수들 초기설정.
        self.all_symbol_df = None
        self.all_symbol_dict = None

        # 업비트 거래소 전체 코인정보 저장.
        self.set_symbol_info()
    
    def set_symbol_info(self):
        # 전체 심볼 먼저 불러와서 저장.
        df = self.get_all_ticker_info()
        rename_dict = {
            'market':'symbol',
            'market_warning':'status',
        }

        df['base_asset'] = df['market'].apply(lambda x: x.split('-')[1])
        df['quote_asset'] = df['market'].apply(lambda x: x.split('-')[0])
        df['adj_symbol'] = df['base_asset'] + "-" + df['quote_asset']
        df.rename(columns=rename_dict, inplace=True)
        # df = df[['adj_symbol', 'symbol', 'base_asset', 'quote_asset', 'status', 'korean_name', 'english_name']]
        df = df[['adj_symbol', 'symbol', 'base_asset', 'quote_asset', 'status']]
        df.set_index('adj_symbol', drop=False, inplace=True)
        self.all_symbol_df = df.copy()
        self.all_symbol_dict = self.all_symbol_df.to_dict('index')

        return

    def get_all_ticker_info(self):
        try:
            url = "https://api.upbit.com/v1/market/all"

            querystring = {"isDetails":"true"} # 유의종목 여부 포함
            headers = {"Accept": "application/json"}
            response = requests.request("GET", url, headers=headers, params=querystring)

            # 전체 코인 정보 Dict로 저장.
            all_symbol_info = json.loads(response.text)
            # 데이터프레임으로 변환하여 저장.
            all_symbol_df = pd.DataFrame(all_symbol_info)
            logging.info(f"Upbit 전체 코인 정보 저장 완료. 총 {all_symbol_df.shape[0]}개.")
            return all_symbol_df
        except:
            logging.error("Upbit 코인정보 API호출 실패.")
            exit(1)

    def get_recent_trades_list(self, adj_symbol, limit=1):
        """주어진 adj_symbol에 대해 가장 최근의 거래(들)을 조회한다.

        Args:
            adj_symbol (int): 이 프로젝트에서 사용하는 코인 심볼. 1개만 사용 가능.
            limit (int, optional): 조회할 거래의 개수. Defaults to 1.
        """
        
        symbol = self.all_symbol_dict[adj_symbol]['symbol']

        url = f"https://api.upbit.com/v1/trades/ticks?market={symbol}&count={str(limit)}"
        headers = {"Accept": "application/json"}
        
        response = requests.get(url, headers=headers)
        recent_trade_list = json.loads(response.text)

        rename_list = []
        for trade in recent_trade_list:
            d = {}
            d['trade_price'] = float(trade['trade_price'])
            d['symbol'] = trade['market']
            d['adj_symbol'] = adj_symbol
            d['base_symbol'] = adj_symbol.split('-')[0]
            d['quote_symbol'] = adj_symbol.split('-')[1]
            d['timestamp'] = int(trade['timestamp'])
            rename_list.append(d)

        return rename_list

    def get_latest_klines_data(self, symbol, interval, latest_n=10, to=''):
        """업비트 코인 캔들 데이터를 업데이트 한다.

        Args:
            symbol (str): 조회하고자 하는 코인 코드(ex. KRW-BTC, BTC-ETH 등)
            interval (str): 얻고자 하는 캔들 데이터의 단위(분 단위). str 형태이어야 함.
            latest_n (int): 조회 데이터 개수(max=200). Defaults to 10.
            to (str, optional): 조회 마지막 시간 지정(exclusive). Defaults to ''.

        Returns:
            pandas dataframe: 조회된 데이터. 오류인 경우 None 리턴.
        """
        # 설정 시간(분)에 맞춰서 url 및 param 저장.
        url = "https://api.upbit.com/v1/candles/minutes/{}".format(interval)
        querystring = {"market":symbol,"count":str(latest_n)}
        headers = {"Accept": "application/json"}
        if to != '':
            querystring['to'] = to

        # API 요청, 결과 변환 및 저장.
        response = requests.request("GET", url, headers=headers, params=querystring)
        response_data = json.loads(response.text)
        response_df = pd.DataFrame(response_data)
        
        # 조회된 데이터가 없는 경우.
        if response_df.shape[0] == 0:
            logging.info("데이터 조회 개수가 0 입니다. 함수 종료.")
            return None

        # 조회된 데이터 개수가 지정한 latest_n 개수와 다른 경우. (오류는 아님)
        if response_df.shape[0] != latest_n:
            logging.info("데이터 조회 개수 불일치(누락 가능성). 요청:{} / 결과:{}".format(latest_n, response_df.shape[0]))

        # 데이터 시작, 끝 시간 저장.
        start_time = response_df['candle_date_time_kst'].min()
        end_time = response_df['candle_date_time_kst'].max()

        # to = start_time + '+09:00'

        logging.info("{} / {} 데이터 요청 완료. {}개. {} ~ {}".format(symbol, interval, response_df.shape[0], start_time, end_time))

        # timestamp에 중복값이 있는지 체크        
        counts = response_df['timestamp'].value_counts()
        duplicate_check = (counts >= 2).sum()
        if duplicate_check > 0:
            logging.info("timestamp가 중복된 값이 있습니다. 데이터를 확인하세요. 함수 종료.")
            return None
        
        if to == "":
            response_df = response_df[:-1].copy()
        
        return response_df


if __name__ == "__main__":
    my_upbit = MyUpbit(api_key=ACCESS_KEY, secret_key=SECRET_KEY)
    # result = MyUpbit.get_latest_klines_data("KRW-BTC", "1", 10)
    result = my_upbit.get_recent_trades_list("BTC-KRW", 1)
    print(result)