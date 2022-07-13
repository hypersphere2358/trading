import logging
import pandas as pd
import numpy as np
import telegram
import os

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
    
    def calculate_symbol_pairs_ratio(self, symbol_pairs_dict):
        # 고려해야 할 것들. (7.13)
        # 차익거래 기회 발생 시, 포지션 진입에는 숏포지션이 필요하다.
        # 즉, 실제 숏포지션 진입 거래를 하기 위해서는 마진(또는 선물)거래의 가격데이터가 필요함.
        # spot 가격으로 계산한 환율을 사용해서 마진거래 진입을 하는 경우 오차가 발생할 수 있을 것 같음.
        # 그래서 마진거래 가격데이터로부터 계산한 환율을 사용해서 마진거래를 해야하지 않을까 싶음.
        
        for base_symbol, secondary_symbol in symbol_pairs_dict.items():
            base_latest_trade = self.base_ex_helper.get_recent_trades_list(adj_symbol=base_symbol)[0]
            secondary_latest_trade = self.secondary_ex_helper.get_recent_trades_list(adj_symbol=secondary_symbol)[0]

            ratio = base_latest_trade['trade_price'] / secondary_latest_trade['trade_price']
            target_symbol = base_latest_trade['base_symbol']
            base_quote = base_latest_trade['quote_symbol']
            secondary_quote = secondary_latest_trade['quote_symbol']
            
            logging.info(f"Target : {target_symbol}, {base_quote}-{secondary_quote} 비율 : {ratio:,.4f}")
        return


if __name__ == "__main__":

    # base 거래소 : 업비트
    # secondary 거래소 : 바이낸스
    # symbol_pairs : {업비트 symbol(adj) : 바이낸스 symbol(adj)} 딕셔너리 형태로 저장.
    # adj_symbol : "BASE_ASSET_TICKER-QUOTE_ASSET_TICKER" 형태의 문자열로, 하이픈(-)으로 구분하여 설정.
    symbol_pairs = {}
    symbol_pairs['BTC-KRW'] = 'BTC-USDT'
    symbol_pairs['ETH-KRW'] = 'ETH-USDT'
    
    trader = ExchangeArbitrager(base_ex="upbit", secondary_ex="binance")
    trader.calculate_symbol_pairs_ratio(symbol_pairs_dict=symbol_pairs)