#%%
# 업비트 API key
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

# 시가총액 상위 10개 코인
KRW_MARKETCAP_TOP_10 = [
    'KRW-BTC',
    'KRW-ETH',
    # 'KRW-ADA',
    # 'KRW-XRP',
    # 'KRW-DOT',
    # 'KRW-DOGE',
    # 'KRW-LINK',
    # 'KRW-LTC',
    # 'KRW-BCH',
    # 'KRW-ATOM',
]

ta.PPO

# 로깅 객체
class MyLog():
    def __init__(self, color_mode='on') -> None:
        self.log_depth = 1
        self.color_mode = color_mode
        
    def depth_plus(self, unit=1):
        self.log_depth += unit
        
    def depth_minus(self, unit=1):
        self.log_depth -= unit
        
    def log_print(self, log_text, plain=False, log_type='info', time=True, pre_linefeed=False, post_linefeed=False):
        """커스텀된 log 문자열을 출력

        Args:
            log_text ([str]): 출력할 문자열
            log_type (str, optional): 'info', 'warning' or 'error' Defaults to 'info'.
            depth (int, optional): 출력 들여쓰기 수준. Defaults to 1.
            time (bool, optional): 시간 출력. Defaults to True.
            pre_linefeed (bool, optional): 출력 전 줄바꿈. Defaults to False.
            post_linefeed (bool, optional): 출력 후 줄바꿈. Defaults to False.
        """
        if plain == False:
            text = ' ' * (self.log_depth-1)*2
            if self.color_mode == 'on':
                if log_type == 'info':      text += '\033[36m' + '[INFO]' + '\033[0m'
                elif log_type == 'error':   text += '\033[31m' + '[ERROR]' + '\033[0m'
                elif log_type == 'warning': text += '\033[33m' + '[WARNING]' + '\033[0m'
                elif log_type == 'important': text += '\033[32m' + '[IMPORTANT]' + '\033[0m'
                else:                       text += '[OTHER]'
            elif self.color_mode == 'off':
                if log_type == 'info':      text += '[INFO]'
                elif log_type == 'error':   text += '[ERROR]'
                elif log_type == 'warning': text += '[WARNING]'
                elif log_type == 'important': text += '[IMPORTANT]'
                else:                       text += '[OTHER]'
                
            if time:
                datetime_str = datetime.datetime.now().strftime("[%m/%d %H:%M:%S]")
                text += datetime_str
            
            text += " " + log_text
        else:
            text = log_text
        
        if pre_linefeed: print()
        print(text)
        if post_linefeed: print()
        
        
# 구글 드라이브 마운트를 위한 exception 클래스.
class GoogleDriveMountError(Exception):
    def __init__(self, message):
        self.message = message

mylog = MyLog()

try:
    # drive 폴더가 없으면 구글 드라이브가 아직 마운트 되지 않은 상태임.
    if os.path.exists(os.path.join(os.getcwd(), 'drive')) == False:
        raise GoogleDriveMountError("Google Drive가 마운트되지 않았습니다.")
    DATABASE_DIR = os.path.join(os.getcwd(), 'drive', 'MyDrive', '코딩', 'Upbit', 'upbit_win', 'data')
    mylog.log_print("Google Drive 마운트 확인.")
except GoogleDriveMountError:
    mylog.log_print("Google Drive 마운트 연결 실패. 로컬 데이터로 연결합니다.", log_type='warning')
    DATABASE_DIR = os.path.join(os.getcwd(), 'data')

# 기초데이터 폴더 경로 지정.
mylog.log_print(f"DATABASE_DIR : {DATABASE_DIR}", post_linefeed=True)

# 업비트 클래스
class UpbitInvestment():
    def __init__(self, data_dir_path) -> None:
        # 업비트에서 거래되고 있는 코인들의 정보를 저장하는 데이터프레임.
        # 객체를 생성하고 사용할 때마다 저장된다.
        self.all_coin_info_df = None
        
        # 코인들의 candle 데이터가 저장되더 있는 dict
        # ex) coindata['KRW-BTC']['5M'] = pandas dataframe
        self.coindata = {}
        self.coindata_update_status = {}
        self.coindata_merged = None
        
        # 메인 데이터 폴더경로 저장.
        self.data_dir_path = data_dir_path
        
        

        mylog.log_print("UpbitInvestment 객체 생성 완료.")
        return


    def coindata_single_data(self, min_unit=5, target_col='trade_price'):
        if len(self.coindata) == 0:
            mylog.log_print("현재 로딩된 코인 데이터가 없습니다.", log_type='warning')
            return

        # 데이터 키 생성.
        unit_key = 'unit_{}_min'.format(min_unit)
        # 가격 데이터를 저장할 dataframe 생성.
        close_df = pd.DataFrame()

        # 현재 저장되어 있는 데이터들의 가격을 모두 merge한다. datetime 값은 합집합.
        for market_code in self.coindata.keys():
            sub_coin_df = self.coindata[market_code][unit_key]
            sub_coin_df = sub_coin_df[[target_col]].copy()
            sub_coin_df.rename(columns={target_col:market_code}, inplace=True)

            close_df = close_df.merge(sub_coin_df, how='outer', left_index=True, right_index=True)
        # 모든 코인 가격 데이터들을 합치더라도, 모든 datetime index가 저장되는 것은 아니다.
        # 즉, datetime index값이 중간중간 비어있는 곳이 있음.
        # 이를 보완하기 위해, 시작, 종료시간을 구하고, 모든 datetime index값을 주기(unit)에 맞춰 생성한 뒤, 코인 가격 데이터를 저장한다.
        datetime_max = close_df.index[-1]
        datetime_min = close_df.index[0]
        # 모든 datetime index 생성.
        datetime_all = pd.date_range(start=datetime_min,
                                     end=datetime_max,
                                     freq=f"{min_unit}T")
        # 모든 datetime index를 갖는 데이터프레임 만들기.
        all_datetime_close_df = pd.DataFrame()
        all_datetime_close_df['temp_col'] = datetime_all
        all_datetime_close_df.index = datetime_all
        all_datetime_close_df.index.name = 'datetime'
        # 가격 데이터 merge
        all_datetime_close_df = all_datetime_close_df.merge(close_df, how='outer', left_index=True, right_index=True)
        all_datetime_close_df.drop(columns='temp_col', inplace=True)

        self.coindata_by_column[target_col] = {}
        self.coindata_by_column[target_col][unit_key] = all_datetime_close_df.copy()
        
        mylog.log_print(f"지정 컬럼만 데이터 merging 완료. target:{target_col} / min:{min_unit}")
    
    def save_coin_data(self, market_code, coin_df):
        filename = market_code + '.csv'
        file_path = os.path.join(DATABASE_DIR, filename)
        coin_df.to_csv(file_path, index=False)
        self.mylog.log_print('{} 데이터파일 저장 완료. {}'.format(market_code, filename))
    
    
    def update_all_coin_candle_data(self, minutes):
        if self.all_coin_info_df is None:
            self.mylog.log_print("전체코인 정보가 없습니다. 정보를 먼저 불러오세요.", log_type='error')
            return False
        
        updated_coin_n = 0
        for i in self.all_coin_info_df.index:
            market_code = self.all_coin_info_df.loc[i, 'market']
            market_warning = self.all_coin_info_df.loc[i, 'market_warning']
            if ('KRW' in market_code) and (market_warning == 'NONE'):
                self.mylog.log_print("코인데이터 조회 - {}".format(market_code), pre_linefeed=True)
                self.get_coin_candle_data(
                    market_code=market_code,
                    minutes=minutes,
                    count=200
                )
                updated_coin_n += 1
        
        self.mylog.log_print("총 {}개 코인 데이터 업데이트 완료.".format(updated_coin_n), pre_linefeed=True)

    def sort_coin_data(self):
        if len(self.coindata) == 0:
            mylog.log_print(f"정렬할 코인 데이터 없음.", log_type='warning')
            return
        
        for market_code in self.coindata.keys():
            coin_dict = self.coindata[market_code]
            for subdata in coin_dict.keys():
                coin_dict[subdata].sort_values(by='candle_date_time_kst', inplace=True)
        mylog.log_print("코인 데이터 정렬 완료.")

    def load_coin_data(self, market_code_list):
        # 정상적으로 로딩 완료된 코인데이터 개수.
        loaded_data_n = 0
        # 전달될 코인 목록 전체 불러오기.
        for market_code in market_code_list:
            # 코인 데이터파일이 있는지 확인.
            if self.check_coin_data_exists(market_code):
                filename = market_code + '.csv'
                file_path = os.path.join(DATABASE_DIR, filename)
                
                # 데이터 파일 불러오기.
                coin_df = pd.read_csv(file_path)
                # 모든 unit 저장.
                unit_list = coin_df['unit'].unique()

                # 해당 코인 데이터를 저장할 dict 생성.
                self.coindata[market_code] = {}
                self.coindata[market_code]['base'] = coin_df.copy()
                for unit in unit_list:
                    unit_key = 'unit_{}_min'.format(unit)
                    sub_coin_df = coin_df[coin_df['unit'] == unit]
                    sub_coin_df.index = sub_coin_df['candle_date_time_kst'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
                    sub_coin_df.index.name = 'datetime'
                    self.coindata[market_code][unit_key] = sub_coin_df.copy()
                
                mylog.log_print(f"{market_code} 코인 데이터 파일 불러오기 완료. {coin_df.shape}")
                loaded_data_n += 1
            else:
                mylog.log_print(f"{market_code} 코인 데이터 파일 없음.")
        self.sort_coin_data()
        return loaded_data_n
    
    def check_coin_data_exists(self, market_code):
        """주어진 코인 market code 데이터 파일이 존재하는지 확인.

        Args:
            market_code ([type]): [description]

        Returns:
            [type]: [description]
        """
        coin_filename = market_code + '.csv'
        return os.path.exists(os.path.join(DATABASE_DIR, coin_filename))
    
    def get_valid_datetime_list(self, df):
        """dataframe 시계열 데이터에서 유효한 데이터들의 시간 값들을 구한다. 유효데이터 시작, 종료, 데이터개수 값을 리턴함.

        Args:
            df ([type]): [description]

        Returns:
            [type]: [description]
        """
        if df.index.dtype != 'datetime64[ns]':
            mylog.log_print("dataframe index의 dtype이 datetime64[ns]가 아닙니다.", log_type='error')
            return
        
        valid_datetime_list = []
        nan_ser = df.isna().sum(axis=1)
        flag = 0
        valid_n = 0
        sub_list = []
        for dt in nan_ser.index:
            if (flag == 0) and (nan_ser[dt] == 0):
                flag = 1
                sub_list.append(dt)
            elif (flag == 1) and (nan_ser[dt] != 0):
                flag = 0
                sub_list.append(dt)
                sub_list.append(valid_n)
                valid_datetime_list.append(sub_list)
                sub_list = []
                valid_n = 0
            
            if flag == 1:
                valid_n += 1

        if flag == 1:
            sub_list.append(np.nan)
            sub_list.append(valid_n)
            valid_datetime_list.append(sub_list)
        return valid_datetime_list
    
    
    
    
    
    
    
# 새로 작성 완료한 method들.

    def generate_input_output_dataframe(self, input_params_dict, output_params_dict):
        """학습에 사용될 입출력 데이터 생성.

        Args:
            input_params_dict (dict): 입력데이터 생성에 사용할 컬럼(key)과 기간 수(value)
            output_params_dict (dict): 출력데이터 생성에 사용할 컬럼(key)과 기간 수(value)
            output_shift (int): 출력(보통 수익률데이터 사용)값 lagging 기간.

        Returns:
            [pandas dataframe]: 입력, 출력값이 포함된 dataframe. 시간 인덱스 포함.
        """
        # 입출력 설정 파라미터로부터 값 저장.
        input_columns = list(input_params_dict.keys())
        input_data_size = sum(input_params_dict.values())
        
        output_column_featured = output_params_dict['featured']
        output_column_original = output_params_dict['original']
        output_shift = output_params_dict['shift_period']
        
        mylog.log_print(f"입출력 데이터 생성 시작. 입력 {len(input_columns)}개. 출력 컬럼 : {output_column_featured}")
        
        # 훈련데이터로 사용할 컬럼만 따로 추출하여 저장. 최근 시간에서 멀어지는 순으로 내림차순 저장.
        input_df = self.coindata_merged[input_columns].copy()
        input_df.sort_index(ascending=False, inplace=True)
        output_df = self.coindata_merged[[output_column_featured, output_column_original]].copy()
        output_df.sort_index(ascending=False, inplace=True)
        output_df = output_df.shift(output_shift)
        
        # 테스트용
        # input_df = input_df[0:100].copy()
        # output_df = output_df[0:100].copy()
        # # display(input_df)
        # display(output_df)
        
        # 데이터프레임 생성용.
        row_list = []
        
        # for i, time_idx in tqdm(enumerate(input_df.index)):
        for i in tqdm(range(input_df.shape[0])):
            row = []
            row.append(input_df.index[i]) # 시간 인덱스 저장.
            for col, period in input_params_dict.items(): # 각 컬럼, 기간수에 맞게 데이터 추출하여 리스트에 펼쳐서 저장.
                data = input_df.loc[input_df.index[i:(i+period)], col].values.tolist()
                row.extend(data)
            
            # 실제 생성되어야 할 입력 데이터 개수와, 생성된 데이터 개수 검증.
            if input_data_size+1 != len(row):
                continue
            
            # 출력 데이터는 1컬럼, 1개값(현재 기준에서의 가정)임.
            data = output_df.loc[output_df.index[i]].values.tolist()
            row.extend(data)
            
            # 생성된 한개 row 데이터 저장.
            row_list.append(row)
        
        # 생성된 데이터를 dataframe으로 저장하고, 출력값 컬럼명 변경.
        input_output_df = pd.DataFrame(row_list)
        input_output_df.set_index(0, drop=True, inplace=True)
        input_output_df.rename(columns={input_output_df.columns[-2]:'output', input_output_df.columns[-1]:'output_original'}, inplace=True)

        mylog.log_print(f"입출력 데이터 생성 완료. 크기 : {input_output_df.shape}")
        print(input_output_df)
        return input_output_df
    
    def generate_feature_ADOSC(self, high, low, close, volume, fastperiod, slowperiod):
        df = self.coindata_merged[[high, low, close, volume]].copy()
        
        adosc = ta.ADOSC(df[high], df[low], df[close], df[volume], fastperiod, slowperiod)
        
        new_colname = volume + "|" + f"ADOSC-{str(fastperiod)}-{str(slowperiod)}"
        df[new_colname] = adosc
        df.drop(columns=[high, low, close, volume], inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def generate_feature_OBV(self, close, volume):
        df = self.coindata_merged[[close, volume]].copy()
        
        obv = ta.OBV(df[close], df[volume])
        
        new_colname = volume + "|" + 'OBV'
        df[new_colname] = obv
        df.drop(columns=[close, volume], inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df

    def generate_feature_DIGITAL(self, target_cols, cut):
        # 계산 적용하려는 컬럼 데이터만 저장.
        df = self.coindata_merged[target_cols].copy()
        for col in df.columns:
            df[col] = pd.cut(df[col], bins=[-99999.0, cut, 99999.0], labels=False)
        
        rename_dict = {}
        for colname in df.columns:
            new_colname = colname + "|" + f"digital-{str(cut)}"
            rename_dict[colname] = new_colname
        df.rename(columns=rename_dict, inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def generate_feature_RSI(self, target_cols, period):
        # 계산 적용하려는 컬럼 데이터만 저장.
        df = self.coindata_merged[target_cols].copy()
        for col in df.columns:
            df[col] = ta.RSI(df[col], period)
        
        rename_dict = {}
        for colname in df.columns:
            new_colname = colname + "|" + f"RSI-{str(period)}"
            rename_dict[colname] = new_colname
        df.rename(columns=rename_dict, inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def generate_feature_pct_change(self, target_cols, period):
        # 계산 적용하려는 컬럼 데이터만 저장.
        df = self.coindata_merged[target_cols].copy()
        df = df.pct_change(period, fill_method=None)
        
        rename_dict = {}
        for colname in df.columns:
            new_colname = colname + "|" + f"ratio-{str(period)}"
            rename_dict[colname] = new_colname
        df.rename(columns=rename_dict, inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def generate_feature_SMA(self, target_cols, period):
        # 계산 적용하려는 컬럼 데이터만 저장.
        df = self.coindata_merged[target_cols].copy()
        df = df.rolling(period).mean()
        
        rename_dict = {}
        for colname in df.columns:
            new_colname = colname + "|" + f"ma-{str(period)}"
            rename_dict[colname] = new_colname
        df.rename(columns=rename_dict, inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def generate_feature_ratio_of_two_cols(self, numerator, denominator):
        # 계산 적용하려는 컬럼 데이터만 저장.
        df = self.coindata_merged[[numerator, denominator]].copy()
        nu = df[numerator].copy()
        de = df[denominator].copy()
        ratio = nu / de - 1
        
        re_numerator = numerator.replace("|", "^")
        re_denominator = denominator.replace("|", "^")
        
        new_colname = f'ratio-({re_numerator})({re_denominator})'
        df[new_colname] = ratio
        df.drop(columns=[numerator, denominator], inplace=True)
        
        df = pd.concat([self.coindata_merged, df], axis=1)
        self.coindata_merged = df.copy()
        return df
    
    def preprocess_merge_candle_data(self, target_cols):
        mylog.log_print('코인데이터 전처리 - Merge.', pre_linefeed=True)
        mylog.depth_plus()
        
        merged_df = pd.DataFrame()
        for market_code in self.coindata.keys():
            for minutes in self.coindata[market_code].keys():
                df = self.coindata[market_code][minutes].copy()
                
                # datetime 인덱스 생성.
                df.index = df['candle_date_time_kst'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
                # 필요한 컬럼만 저장.
                df = df[target_cols]
                
                rename_dict = {}
                for colname in target_cols:
                    new_colname = "|".join([market_code, minutes, colname])
                    rename_dict[colname] = new_colname
                    
                df.rename(columns=rename_dict, inplace=True)
                
                if merged_df.shape == (0, 0):
                    merged_df = df.copy()
                else:
                    merged_df = merged_df.merge(df, how='outer', left_index=True, right_index=True)
                
                # start_time = df['candle_date_time_kst'].min()
                # end_time = df['candle_date_time_kst'].max()
                
                # mylog.log_print(f'{market_code} / {minutes} 데이터. 크기 {df.shape}. {start_time} ~ {end_time}')
        mylog.depth_minus()
        
        self.coindata_merged = merged_df.sort_index(ascending=True).copy()
        return self.coindata_merged
        
    def check_coin_candle_data(self):
        mylog.log_print('코인데이터 정보 체크.', pre_linefeed=True)
        mylog.depth_plus()
        for market_code in self.coindata.keys():
            for minutes in self.coindata[market_code].keys():
                df = self.coindata[market_code][minutes]
                
                start_time = df['candle_date_time_kst'].min()
                end_time = df['candle_date_time_kst'].max()
                
                mylog.log_print(f'{market_code} / {minutes} 데이터. 크기 {df.shape}. {start_time} ~ {end_time}')
        mylog.depth_minus()
        
    def save_all_coin_candle_data(self, only_updated=False):
        
        for market_code in self.coindata.keys():
            for minutes in self.coindata[market_code].keys():
                filename = minutes + ".csv"
                coin_min_file_path = os.path.join(self.data_dir_path, 'candle_data', market_code, filename)
                self.coindata[market_code][minutes].to_csv(coin_min_file_path, index=False)
                
                mylog.log_print(f'{market_code} / {minutes} 데이터 파일 저장 완료. {coin_min_file_path}')        
    
    def update_coin_candle_data_backward(self, market_code, minutes, count):
        
        
        # 코인 데이터 사본을 저장한다.
        try:
            coin_df = self.coindata[market_code][minutes].copy()
        except: # 없는 경우 에러
            mylog.log_print(f'로드되어 있는 {market_code} / {minutes} 코인 데이터가 없습니다. 함수 종료.', log_type='error')
            return None
        
        # 업데이트 전 시작, 끝 시간 저장.
        cur_start_time = coin_df['candle_date_time_kst'].min()
        cur_end_time = coin_df['candle_date_time_kst'].max()
        mylog.log_print(f'{market_code} / {minutes} 기존 데이터 : {cur_start_time} ~ {cur_end_time}')
        
        to = cur_start_time + '+09:00' # 최초에는 가장 현재 데이터의 시작 시점까지로 조회한다.
        count_step = 200    # 매 조회시 최대 조회 수
        update_df = pd.DataFrame()
        while count > 0:
            if count < count_step:
                count_step = count
            
            # 데이터 request.
            response_df = self.get_coin_candle_data(market_code, minutes, count_step, to)
            if response_df is None:
                mylog.log_print('데이터 request 오류. 함수 종료.', log_type='error')
                return None
            
            # 조회 카운트 개수 감소.
            count -= count_step
            update_df = update_df.append(response_df)
            
            # 데이터 정상 조회 완료 시.
            update_start_time = update_df['candle_date_time_kst'].min()
            update_end_time = update_df['candle_date_time_kst'].max()
                        
            # 반복될 다음 조회시에는 방금 조회한 데이터의 처음 시간까지로 조회한다.
            to = update_start_time + '+09:00'            
            time.sleep(0.2) # 시간 지연
        
        update_start_time = update_df['candle_date_time_kst'].min()
        update_end_time = update_df['candle_date_time_kst'].max()
        mylog.log_print(f'추가할 데이터 : {update_start_time} ~ {update_end_time}')
        
        # 데이터 추가.
        coin_df = coin_df.append(update_df)
        coin_df.reset_index(drop=True, inplace=True)
        cur_start_time = coin_df['candle_date_time_kst'].min()
        cur_end_time = coin_df['candle_date_time_kst'].max()
        
        # 객체변수에 다시 저장.
        self.coindata[market_code][minutes] = coin_df.copy()
        mylog.log_print(f'추가완료 후 데이터 : {cur_start_time} ~ {cur_end_time}')
    
    def update_coin_candle_data_forward(self, market_code, minutes, count=None):
        
        
        # 코인 데이터 사본을 저장한다.
        try:
            coin_df = self.coindata[market_code][minutes].copy()
        except: # 없는 경우 에러
            mylog.log_print(f'로드되어 있는 {market_code} / {minutes} 코인 데이터가 없습니다. 함수 종료.', log_type='error')
            return None
        
        # 업데이트 전 시작, 끝 시간 저장.
        cur_start_time = coin_df['candle_date_time_kst'].min()
        cur_end_time = coin_df['candle_date_time_kst'].max()
        mylog.log_print(f'{market_code} / {minutes} 기존 데이터 : {cur_start_time} ~ {cur_end_time}')
        
        to = '' # 최초에는 가장 최근 시점 데이터까지로 조회한다.
        update_df = pd.DataFrame()
        while True:
            # 데이터 request.
            response_df = self.get_coin_candle_data(market_code, minutes, 200, to)
            if response_df is None:
                mylog.log_print('데이터 request 오류. 함수 종료.', log_type='error')
                return None
            
            update_df = update_df.append(response_df)
            
            # 데이터 정상 조회 완료 시.
            update_start_time = update_df['candle_date_time_kst'].min()
            update_end_time = update_df['candle_date_time_kst'].max()
            
            # 기존 데이터의 마지막 시간이 추가하는 데이터의 처음 시간보다 같거나 큰 경우. 업데이트를 그만해도 됨.
            if cur_end_time >= update_start_time:
                break
            
            # 반복될 다음 조회시에는 방금 조회한 데이터의 처음 시간까지로 조회한다.
            to = update_start_time + '+09:00'
            time.sleep(0.2) # 시간 지연
        
        
        # 기존 데이터 이후의 데이터만 다시 골라낸다.
        update_df = update_df[update_df['candle_date_time_kst'] > cur_end_time].copy()
        update_start_time = update_df['candle_date_time_kst'].min()
        update_end_time = update_df['candle_date_time_kst'].max()
        mylog.log_print(f'추가할 데이터 : {update_start_time} ~ {update_end_time}')
        
        # 데이터 추가.
        coin_df = update_df.append(coin_df)
        coin_df.reset_index(drop=True, inplace=True)
        cur_start_time = coin_df['candle_date_time_kst'].min()
        cur_end_time = coin_df['candle_date_time_kst'].max()
        
        # 객체변수에 다시 저장.
        self.coindata[market_code][minutes] = coin_df.copy()
        mylog.log_print(f'추가완료 후 데이터 : {cur_start_time} ~ {cur_end_time}')
    
    def load_coins_candle_data(self, market_code_list, minutes):
        """코인들의 캔들 데이터 파일을 불러온다. (기준 : 코인코드, 분(minutes)) 데이터가 없을 경우에는 불러오지 않는다.

        Args:
            market_code_list (list of str): 로드하려고 하는 코인들의 코드 리스트.
            minutes (str): 기준 분. '5', '1' 등과 같이 문자열로 전달해야 한다.
        """
        # 실제 데이터파일로 사용할 파일명
        filename = minutes + ".csv"
        
        # 코드 전체에 대해 처리.
        for market_code in market_code_list:
            # 코인별 폴더 경로, 코인별 분봉 데이터파일 경로
            coin_dir_path = os.path.join(self.data_dir_path, 'candle_data', market_code)
            coin_min_file_path = os.path.join(self.data_dir_path, 'candle_data', market_code, filename)
            
            # 폴더, 파일 존재여부 체크.
            coin_dir_check = os.path.isdir(coin_dir_path)
            coin_min_file_check = os.path.exists(coin_min_file_path)
            
            # 파일이 없는 경우.
            if coin_min_file_check == False:
                mylog.log_print(f'{market_code} / {minutes} 캔들 데이터 없음. 업데이트를 시작합니다.')
                
                # 코인 폴더가 없는 경우, 폴더 생성부터 해준다.
                if coin_dir_check==False:
                    os.mkdir(coin_dir_path)
                    mylog.log_print(f'{coin_dir_path} 폴더 생성 완료.')
                    
                # 폴더는 만들어졌으므로 데이터 생성. 가장 최근 시점까지 200개 생성이 기본.
                coin_df = self.get_coin_candle_data(market_code, minutes, 200)
                coin_df.to_csv(coin_min_file_path, index=False)
            # 데이터파일이 있는 경우. 파일을 불러와서 저장한다.
            else:
                coin_df = pd.read_csv(coin_min_file_path)
            
            # 객체의 코인데이터에 해당 코인 데이터가 없는 경우에는 딕셔너리 새로 생성.
            if self.coindata.get(market_code) is None:
                self.coindata[market_code] = {}
            # 객체에 코인데이터 저장.
            self.coindata[market_code][minutes] = coin_df.copy()
            
        return
    
    def get_coin_candle_data(self, market_code, minutes, count=200, to=''):
        """코인 캔들 데이터를 업데이트 한다.

        Args:
            market_code (str): 조회하고자 하는 코인 코드
            minutes (str): 캔들 기준 분. str 형태이어야 함.
            count (int): 조회 데이터 개수(max=200). Defaults to 200.
            to (str, optional): 조회 마지막 시간 지정(exclusive). Defaults to ''.

        Returns:
            pandas dataframe: 조회된 데이터. 오류인 경우 None 리턴.
        """
            
        # 설정 시간(분)에 맞춰서 url 및 param 저장.
        url = "https://api.upbit.com/v1/candles/minutes/{}".format(minutes)
        querystring = {"market":market_code,"count":str(count)}
        headers = {"Accept": "application/json"}
        if to != '':
            querystring['to'] = to

        # API 요청, 결과 변환 및 저장.
        response = requests.request("GET", url, headers=headers, params=querystring)
        response_data = json.loads(response.text)
        response_df = pd.DataFrame(response_data)
        
        # 조회된 데이터가 없는 경우.
        if response_df.shape[0] == 0:
            mylog.log_print("데이터 조회 개수가 0 입니다. 함수 종료.", log_type='error')
            return None

        # 조회된 데이터 개수가 지정한 count 개수와 다른 경우. (오류는 아님)
        if response_df.shape[0] != count:
            mylog.log_print("데이터 조회 개수 불일치(누락 가능성). 요청:{} / 결과:{}".format(count, response_df.shape[0]), log_type='warning')

        # 데이터 시작, 끝 시간 저장.
        start_time = response_df['candle_date_time_kst'].min()
        end_time = response_df['candle_date_time_kst'].max()

        # to = start_time + '+09:00'

        mylog.log_print("{} / {} 데이터 요청 완료. {}개. {} ~ {}".format(market_code, minutes, response_df.shape[0], start_time, end_time))

        # timestamp에 중복값이 있는지 체크        
        counts = response_df['timestamp'].value_counts()
        duplicate_check = (counts >= 2).sum()
        if duplicate_check > 0:
            mylog.log_print("timestamp가 중복된 값이 있습니다. 데이터를 확인하세요. 함수 종료.", log_type='error')
            return None
        
        if to == "":
            response_df = response_df[:-1].copy()
        
        return response_df
    
    def get_all_coins_info(self):
        """(21.12.20 완료 v1) 업비트 거래소에서 거래되는 모든 코인들의 정보를 저장한다.

        Returns:
            bool: 불러오기 결과.
        """
        try:
            url = "https://api.upbit.com/v1/market/all"

            querystring = {"isDetails":"true"} # 유의종목 여부 포함
            headers = {"Accept": "application/json"}
            response = requests.request("GET", url, headers=headers, params=querystring)

            # 전체 코인 정보 Dict로 저장.
            all_coin_info = json.loads(response.text)
            # 데이터프레임으로 변환하여 저장.
            self.all_coin_info_df = pd.DataFrame(all_coin_info)
            mylog.log_print(f"전체 코인정보 저장 완료. 총 {self.all_coin_info_df.shape[0]}개")
            print(self.all_coin_info_df.head())
            print()
            return True
        except:
            self.mylog.log_print("전체 코인정보 API호출 실패.", log_type='error')
            return False


def generate_input_state_number(input_data, base):
    input_data = [str(int(x)) for x in input_data]
    input_data_str = "".join(input_data)
    input_data_state = int(input_data_str, base)
    return input_data_state
    
#%%

upbit = UpbitInvestment(data_dir_path=DATABASE_DIR)
# upbit.get_all_coins_info()
upbit.load_coins_candle_data(['KRW-BTC', 'KRW-ETH', 'KRW-ADA'], '5')

upbit.update_coin_candle_data_backward('KRW-BTC', '5', 5000)
upbit.update_coin_candle_data_backward('KRW-ETH', '5', 5000)
upbit.update_coin_candle_data_backward('KRW-ADA', '5', 5000)

upbit.save_all_coin_candle_data()

upbit.check_coin_candle_data()

#
upbit.preprocess_merge_candle_data(['trade_price', 'high_price', 'low_price', 'candle_acc_trade_volume'])

coin_list = ['KRW-BTC', 'KRW-ETH', 'KRW-ADA']

for c in coin_list:
    upbit.generate_feature_pct_change([f'{c}|5|trade_price'], 1)
    upbit.generate_feature_pct_change([f'{c}|5|trade_price'], 3)
    upbit.generate_feature_pct_change([f'{c}|5|trade_price'], 5)
    upbit.generate_feature_pct_change([f'{c}|5|candle_acc_trade_volume'], 1)
    upbit.generate_feature_SMA([f'{c}|5|trade_price'], 20)
    upbit.generate_feature_SMA([f'{c}|5|candle_acc_trade_volume'], 20)
    upbit.generate_feature_ratio_of_two_cols(f'{c}|5|trade_price', f'{c}|5|trade_price|ma-20')
    upbit.generate_feature_ratio_of_two_cols(f'{c}|5|candle_acc_trade_volume', f'{c}|5|candle_acc_trade_volume|ma-20')
    upbit.generate_feature_ratio_of_two_cols(f'{c}|5|trade_price', f'{c}|5|low_price')
    upbit.generate_feature_ratio_of_two_cols(f'{c}|5|high_price', f'{c}|5|trade_price')
    upbit.generate_feature_ratio_of_two_cols(f'{c}|5|high_price', f'{c}|5|low_price')


target_coin = 'KRW-ETH'
cut_value = 0.001

upbit.generate_feature_DIGITAL([f'{target_coin}|5|trade_price|ratio-1'], cut_value)


input_params = {}
for c in coin_list:
    input_params[f'{c}|5|trade_price|ratio-1'] = 5
    input_params[f'{c}|5|trade_price|ratio-3'] = 1
    input_params[f'{c}|5|trade_price|ratio-5'] = 1
    input_params[f'{c}|5|candle_acc_trade_volume|ratio-1'] = 5
    input_params[f'ratio-({c}^5^trade_price)({c}^5^trade_price^ma-20)'] = 5
    input_params[f'ratio-({c}^5^candle_acc_trade_volume)({c}^5^candle_acc_trade_volume^ma-20)'] = 5
    input_params[f'ratio-({c}^5^trade_price)({c}^5^low_price)'] = 5
    input_params[f'ratio-({c}^5^high_price)({c}^5^trade_price)'] = 5
    input_params[f'ratio-({c}^5^high_price)({c}^5^low_price)'] = 5

output_params = {}
output_params['featured'] = f'KRW-ETH|5|trade_price|ratio-1|digital-{cut_value}'
output_params['original'] = 'KRW-ETH|5|trade_price|ratio-1'
output_params['shift_period'] = 1

input_output_df = upbit.generate_input_output_dataframe(input_params, output_params)

input_output_df.dropna(inplace=True)
input_df = input_output_df.drop(columns=['output', 'output_original'])

X = input_df.values
y = input_output_df['output'].values



test_n = 100000
X_test = X[:test_n,]
y_test = y[:test_n]
X_train = X[test_n:,]
y_train = y[test_n:]

clf = RandomForestClassifier(max_depth=12, random_state=0)
clf.fit(X_train, y_train)

pred_y = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, pred_y)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.show()

#%%

input_data_size = input_df.shape[1]
state_n = generate_input_state_number([1]*input_data_size, 2) + 1

print(f"입력데이터 크기 : {input_data_size}")
print(f"입력데이터 개수 : {input_df.shape[0]}")
print(f"가능한 state 전체 개수 : {state_n}")

state_count_dict = {}
for i in range(state_n):
    state_count_dict[i] = 0
for idx in input_df.index:
    input_data = input_df.loc[idx].values.tolist()    
    state = generate_input_state_number(input_data, 2)
    
    # print(f"{input_data} / {state}")
    state_count_dict[state] += 1
    
state_count_df = pd.DataFrame.from_dict(state_count_dict, orient='index')
plt.bar(x=state_count_df.index, height=state_count_df[0])
# df = upbit.generate_feature_pct_change(['KRW-BTC|5|trade_price'], 6)
# df = upbit.generate_feature_SMA(['KRW-BTC|5|trade_price'], 2)
# df = upbit.generate_feature_ratio_of_two_cols('KRW-BTC|5|trade_price', 'KRW-ETH|5|trade_price')
# df = upbit.generate_feature_OBV(close='KRW-BTC|5|trade_price', volume='KRW-BTC|5|candle_acc_trade_volume')
# df = upbit.generate_feature_ADOSC(high='KRW-BTC|5|high_price', low='KRW-BTC|5|low_price', close='KRW-BTC|5|trade_price', volume='KRW-BTC|5|candle_acc_trade_volume', fastperiod=3, slowperiod=10)
# df = upbit.generate_feature_RSI(['KRW-BTC|5|trade_price'], 2)
# print(df)
# df.to_csv('sample_data.csv')

#%%

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x

class Env():
    def __init__(self, input_output_df, input_size):
        # 오름차순으로 데이터 정렬.
        self.x = input_output_df.sort_index(ascending=True).copy()
        self.input_size = copy.deepcopy(input_size)
        
        # self.x = pd.merge(self.featured_df, self.original_df, left_index=True, right_index=True, how='outer')
        # for i, c in enumerate(self.x.columns):
        #     self.x[i] = self.x[c].pct_change()
        # self.x = self.x.loc[self.x.index > '2017-01-01']        

    def __iter__(self):
        return iter(range(len(self.x)))
    
    def observe_return(self, i):
        if i <len(self.x):
            ret = self.x.loc[self.x.index[i], 'output_original']
        else:
            ret = None
        return ret
            
    def observe_state(self, i, success):
        if i < len(self.x):
            input_value_list = list(self.x.loc[self.x.index[i]].iloc[:self.input_size])
            if np.any(pd.isna(input_value_list)):
                state = None
                input_value_list = None
            else:
                input_value_list.append(success)
                state = generate_input_state_number(input_value_list, 2)
        else:
            state = None
            input_value_list = None
        return (state, input_value_list)
    
    def reward(self, i_timestep, action):
        output = self.x.loc[self.x.index[i_timestep], 'output']
        if action == output:
            return 1
        else:
            return -1
    
class Agent():
    def __init__(self, n_states, n_actions, gamma):
        self.q = np.ones([n_states, n_actions]) / 10
        self.q[:,1] = 0
        self.gamma = gamma
        self.eps = 1e-8
    
    def action(self, state, method='egreedy'):
        if state is None:
            return None
        else:
            value_table = self.value()
            prob = value_table[state,:]
            if method == 'egreedy':
                i_action = np.random.choice(np.arange(2),1,p=prob).item()
            elif method == 'inference':
                i_action = np.argmax(prob)
            return (state, i_action,)
    
    def update(self, state_action, next_state_action, reward):
        next_q = 0 if next_state_action is None else self.q[next_state_action]
        self.q[state_action] = copy.deepcopy(reward + self.gamma*next_q)

    def value(self):
        return softmax(self.q)

env = Env(input_output_df, input_data_size)
agent = Agent(state_n*2, 2, 0.5)

#%%

train_epoch = 10
# action_table = np.array([True,False]) # buy, hold
# best_reward = -100
# count = 0
# max_count = 10
# while count < max_count:
for e in range(train_epoch):
    print(f"epoch {e}")
    pnl = [100]
    bm = [100]
    position = False
    previous_price = None
    reward = 0
    
    success = False
    for i in tqdm(env):
        now_state = env.observe_state(i, success)
        now_state_action = agent.action(now_state[0], method='egreedy')
        

        reward = env.reward(i, now_state_action[1])
        if reward > 0:
            success = True
        else:
            success = False
        
        next_state = env.observe_state(i+1, success) 
        next_state_action = agent.action(next_state[0], method='inference')

        agent.update(now_state_action, next_state_action, reward)

pred = tuple()
pnl = (100.,)
bm = (100.,)

success = False
for i in env:
    now_state = env.observe_state(i, success)
    now_state_action = agent.action(now_state[0], method='inference')
    
    pred += (now_state_action[1], )
    
    if len(pred) > 0:
        pnl += (pnl[-1]*((1+env.observe_return(i)) if pred[-1]==1 else 1),)
        bm += (bm[-1]*(1+env.observe_return(i)),)

    reward = env.reward(i, now_state_action[1])
    if reward > 0:
        success = True
    else:
        success = False

print(f"bm : {bm[-1]}, pnl : {pnl[-1]}")

#%% 
# upbit.load_coin_data(KRW_MARKETCAP_TOP_10)
# upbit.coindata_single_data(target_col='trade_price')
# upbit.coindata_single_data(target_col='opening_price')
# upbit.coindata_single_data(target_col='high_price')
# upbit.coindata_single_data(target_col='low_price')
# upbit.coindata_single_data(target_col='candle_acc_trade_price')
# upbit.coindata_single_data(target_col='candle_acc_trade_volume')

# params_dict = {}
# params_dict['inputs'] = [
#     ['trade_price', 'unit_5_min', 'pct_change', 1, 5],
#     ['opening_price', 'unit_5_min', 'pct_change', 1, 5],
#     ['high_price', 'unit_5_min', 'pct_change', 1, 5],
#     ['low_price', 'unit_5_min', 'pct_change', 1, 5],
#     ['candle_acc_trade_price', 'unit_5_min', 'pct_change', 1, 5],
#     ['candle_acc_trade_volume', 'unit_5_min', 'pct_change', 1, 5],
# ]
# params_dict['output'] = [
#     'trade_price', 'unit_5_min', 'KRW-ETH', 'pct_change', 1
# ]

# result = upbit.generate_input_output_dataframe(params_dict=params_dict)
# # display(result)

# re_df = result.copy()
# re_df.dropna(inplace=True)
# re_df['output'] = re_df['output'].apply(lambda x: 1 if x > 0.0005 else 0)
# display(re_df)
# re_df.to_csv('sample_dataset.csv', index=False)
# # upbit.get_valid_datetime_list(upbit.coindata_by_column['trade_price']['unit_5_min'])
# # %%
