import logging
import re
from unittest import result
import pandas as pd
import numpy as np
import telegram
import os
import time
import datetime

import helper_binance as bi
import helper_upbit as up

class Trader():
    def __init__(self):
        return

class TechnicalAnalyser(Trader):
    def __init__(self):
        super().__init__()

class Arbitrager(Trader):
    
    def __init__(self):
        super().__init__()

class ExchangeArbitrager(Arbitrager):
    ################################################################################
    # 고려해야 할 것들.
    # - 바이낸스는 같은 BTC라 하더라도, USDC, USDT, BUSD 등 quote가 되는 기준 스테이블 코인이 다르고, 각각 정확히 $1가 아님.
    # - 이를 고려하기 위해, 위 세개 스테이블 코인 기준 가격을 모두 조회하여 평균을 구해 업비트 KRW-BTC 가격과 비교해야 좀 더 정확하지 않을까 하는 생각.
    def __init__(self, base_ex, secondary_ex):
        super().__init__()
        logging.info(f"Initializing {self.__class__.__name__}.")

        # 거래소 변수 초기화.
        logging.info(f"Base 거래소 설정 - {base_ex}.")
        self.set_base_exchange(exchange=base_ex)

        logging.info(f"Secondary 거래소 설정 - {secondary_ex}.")
        self.set_secondary_exchange(exchange=secondary_ex)

        # 거래소 설정이 정상적으로 완료되지 않은 경우.
        if self.base_ex_helper is None or self.secondary_ex_helper is None:
            logging.error("Base, secondary 거래소 설정 오류.")
            exit(1)

    
    def set_base_exchange(self, exchange):

        if exchange == "upbit":
            self.base_ex_helper = up.MyUpbit(up.ACCESS_KEY, up.SECRET_KEY)
        else:
            logging.critical(f"Invalid exchange : {exchange}")
            self.base_ex_helper = None
        return
    
    def set_secondary_exchange(self, exchange):

        if exchange == "binance":
            self.secondary_ex_helper = bi.MyBinance(bi.API_KEY, bi.SECRET_KEY)
        else:
            logging.critical(f"Invalid exchange : {exchange}")
            self.secondary_ex_helper = None
        return
    
    def get_market_data(self, symbol_pairs_dict, trade_recent_n=10):
        # 결과들을 저장할 리스트
        result_list = []

        # 모든 베이스-세컨더리 심볼 페어에 대해 각각 조회.
        for base_ex_symbol, secondary_ex_symbol in symbol_pairs_dict.items():
            ################################################################################################
            # 현물 거래 데이터
            # recent_n개 만큼의 최근 거래 데이터 조회.
            base_ex_recent_trade_list = self.base_ex_helper.get_spot_recent_trades_by_adj_symbol(adj_symbol=base_ex_symbol, limit=trade_recent_n)
            secondary_ex_recent_trade_list = self.secondary_ex_helper.get_spot_recent_trades_by_adj_symbol(adj_symbol=secondary_ex_symbol, limit=trade_recent_n)
            # 최근 거래들을 dataframe 형태로 저장.
            base_ex_recent_trade_df = pd.DataFrame.from_dict(base_ex_recent_trade_list)
            secondary_ex_recent_trade_df = pd.DataFrame.from_dict(secondary_ex_recent_trade_list)

            # 가장 최근의 거래 데이터는 별도 저장.
            base_ex_latest_trade = base_ex_recent_trade_list[0]
            secondary_ex_latest_trade = secondary_ex_recent_trade_list[0]
            
            # 현물 최근 거래가격의 가중평균 계산.
            base_ex_quote_qty_wavg = np.average(base_ex_recent_trade_df['trade_price'], weights=base_ex_recent_trade_df['trade_volume'])
            secondary_ex_quote_qty_wavg = np.average(secondary_ex_recent_trade_df['trade_price'], weights=secondary_ex_recent_trade_df['trade_volume'])
            
            # 필요한 데이터들 계산 및 dict에 저장.
            exchange_ratio = base_ex_quote_qty_wavg / secondary_ex_quote_qty_wavg
            vehicle_symbol = base_ex_latest_trade['base_symbol']
            base_ex_quote = base_ex_latest_trade['quote_symbol']
            base_ex_trade_price = base_ex_latest_trade['trade_price']
            secondary_ex_quote = secondary_ex_latest_trade['quote_symbol']
            secondary_ex_trade_price = secondary_ex_latest_trade['trade_price']

            result_dict = {}
            result_dict['vehicle_symbol'] = vehicle_symbol
            result_dict['base_ex_quote'] = base_ex_quote
            result_dict['base_ex_trade_price'] = base_ex_trade_price
            result_dict['secondary_ex_quote'] = secondary_ex_quote
            result_dict['secondary_ex_trade_price'] = secondary_ex_trade_price
            result_dict['exchange_ratio'] = exchange_ratio
            result_dict['timestamp'] = int(time.mktime(datetime.datetime.now().timetuple())*1000)
            result_list.append(result_dict)
            
            # logging.info(f"Target : {vehicle_symbol}, {base_ex_quote}-{secondary_ex_quote} 비율 : {exchange_ratio:,.4f} (최근 {trade_recent_n}개)")
        return result_list

    def calculate_symbol_pairs_ratio(self, symbol_pairs_dict, trade_recent_n=10):
        # 고려해야 할 것들. (7.13)
        # 차익거래 기회 발생 시, 포지션 진입에는 숏포지션이 필요하다.
        # 즉, 실제 숏포지션 진입 거래를 하기 위해서는 마진(또는 선물)거래의 가격데이터가 필요함.
        # spot 가격으로 계산한 환율을 사용해서 마진거래 진입을 하는 경우 오차가 발생할 수 있을 것 같음.
        # 그래서 마진거래 가격데이터로부터 계산한 환율을 사용해서 마진거래를 해야하지 않을까 싶음.
        
        result_list = []
        for base_ex_symbol, secondary_ex_symbol in symbol_pairs_dict.items():
            base_ex_recent_trade_list = self.base_ex_helper.get_spot_recent_trades_by_adj_symbol(adj_symbol=base_ex_symbol, limit=trade_recent_n)
            secondary_ex_recent_trade_list = self.secondary_ex_helper.get_spot_recent_trades_by_adj_symbol(adj_symbol=secondary_ex_symbol, limit=trade_recent_n)

            base_ex_latest_trade = base_ex_recent_trade_list[0]
            secondary_ex_latest_trade = secondary_ex_recent_trade_list[0]

            base_ex_recent_trade_df = pd.DataFrame.from_dict(base_ex_recent_trade_list)
            secondary_ex_recent_trade_df = pd.DataFrame.from_dict(secondary_ex_recent_trade_list)

            base_ex_quote_qty_wavg = np.average(base_ex_recent_trade_df['trade_price'], weights=base_ex_recent_trade_df['trade_volume'])
            secondary_ex_quote_qty_wavg = np.average(secondary_ex_recent_trade_df['trade_price'], weights=secondary_ex_recent_trade_df['trade_volume'])
            

            exchange_ratio = base_ex_quote_qty_wavg / secondary_ex_quote_qty_wavg
            vehicle_symbol = base_ex_latest_trade['base_ex_symbol']
            base_ex_quote = base_ex_latest_trade['quote_symbol']
            secondary_ex_quote = secondary_ex_latest_trade['quote_symbol']

            result_dict = {}
            result_dict['vehicle_symbol'] = vehicle_symbol
            result_dict['base_ex_quote'] = base_ex_quote
            result_dict['secondary_ex_quote'] = secondary_ex_quote
            result_dict['exchange_ratio'] = exchange_ratio
            result_dict['timestamp'] = int(time.mktime(datetime.datetime.now().timetuple())*1000)
            result_list.append(result_dict)
            
            # logging.info(f"Target : {vehicle_symbol}, {base_ex_quote}-{secondary_ex_quote} 비율 : {exchange_ratio:,.4f} (최근 {trade_recent_n}개)")
        return result_list


if __name__ == "__main__":

    # base 거래소 : 업비트
    # secondary 거래소 : 바이낸스
    # symbol_pairs : {업비트 symbol(adj) : 바이낸스 symbol(adj)} 딕셔너리 형태로 저장.
    # adj_symbol : "BASE_ASSET_TICKER-QUOTE_ASSET_TICKER" 형태의 문자열로, 하이픈(-)으로 구분하여 설정.
    symbol_pairs = {}
    symbol_pairs['BTC-KRW'] = 'BTC-USDT'
    symbol_pairs['ETH-KRW'] = 'ETH-USDT'
    symbol_pairs['XRP-KRW'] = 'XRP-USDT'
    symbol_pairs['ADA-KRW'] = 'ADA-USDT'
    symbol_pairs['SOL-KRW'] = 'SOL-USDT'
    symbol_pairs['DOGE-KRW'] = 'DOGE-USDT'
    symbol_pairs['DOT-KRW'] = 'DOT-USDT'
    symbol_pairs['MATIC-KRW'] = 'MATIC-USDT'
    symbol_pairs['AVAX-KRW'] = 'AVAX-USDT'
    symbol_pairs['TRX-KRW'] = 'TRX-USDT'
    
    trader = ExchangeArbitrager(base_ex="upbit", secondary_ex="binance")
    total_result_list = []
    cnt = 0
    filename = "trading_data.csv"
    while True:
        re = trader.get_market_data(symbol_pairs_dict=symbol_pairs)
        total_result_list.extend(re)
        cnt += 1
        logging.info(f"{cnt} 처리완료. 결과 {len(re)}개.")
        if cnt % 1 == 0:
            result_df = pd.DataFrame.from_dict(total_result_list)
            result_df.to_csv(filename)
            logging.info(f"► 결과파일 중간저장 완료. 크기 : {result_df.shape}")
        
        time.sleep(5)
    print(pd.DataFrame.from_dict(total_result_list))
