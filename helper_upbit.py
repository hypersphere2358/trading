# 업비트 API key(2022.06.30)
access_key = "YtozFZ6R5lgg3gO0ZhxaJpfX5X8avw1PEJShisVF"
secret_key = "c6A9yadQVob87xnwBXbuVzafxcDiMb3rOv3hozuK"

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

class myUpbit():
    def __init__(self):
        return
    
    def get_recent_trades(self, symbol, limit=1):
        """주어진 symbol에 대해 가장 최근의 거래(들)을 조회한다.

        Args:
            symbol (int): 업비트에서 사용하는 코인 코드. 1개만 사용 가능.
            limit (int, optional): 조회할 거래의 개수. Defaults to 1.
        """
        

        url = f"https://api.upbit.com/v1/trades/ticks?market={symbol}&count={str(limit)}"
        headers = {"Accept": "application/json"}
        
        response = requests.get(url, headers=headers)
        recent_trade_list = json.loads(response.text)

        return recent_trade_list

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
    myupbit = myUpbit()
    # result = myupbit.get_latest_klines_data("KRW-BTC", "1", 10)
    result = myupbit.get_recent_trades("KRW-BTC", 1)
    print(result)