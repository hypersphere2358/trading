import pandas as pd
import numpy as np
import telegram
import os

import helper_binance as bi
import helper_upbit as up

class trader():
    def __init__(self):
        return

class technicalAnalyser(trader):
    def __init__(self):
        super().__init__()

class arbitrager(trader):
    
    def __init__(self):
        super().__init__()

class exchangeArbitrager(arbitrager):
    ################################################################################
    # 고려해야 할 것들.
    # - 바이낸스는 같은 BTC라 하더라도, USDC, USDT, BUSD 등 quote가 되는 기준 스테이블 코인이 다르고, 각각 정확히 $1가 아님.
    # - 이를 고려하기 위해, 위 세개 스테이블 코인 기준 가격을 모두 조회하여 평균을 구해 업비트 KRW-BTC 가격과 비교해야 좀 더 정확하지 않을까 하는 생각.
    def __init__(self):
        super().__init__()
    
    def set_base_exchange(self):
        return
    
    def set_foreign_exchange(self):
        return


if __name__ == "__main__":
    t = technicalAnalyser()