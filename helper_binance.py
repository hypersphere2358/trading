from hashlib import new
from binance.client import Client

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

###################################################################################################################
# 바이낸스 api 정보
API_KEY = "FTSCvKrWGQeHUvGTO7IEQFIR0g7lteYNCerlowBfiVT1CvEHtrJ2yveBGg0OZdfc"
SECRET_KEY = "87Ja6V7L6tiwHAGmFjwOgHvF4bdHiC7WOW9r59k2CfHPy58nSSPrZ77Oj7FooOBy"

# 텔레그램 api 정보
TELEGRAM_BOT_TOKEN = "5346819653:AAELZmh2hQfG6SC0ZGT2ftKYwk4gqIthySU"
TELEGRAM_CHAT_ID = "-1001669316681"


###################################################################################################################
tele_bot = telegram.Bot(TELEGRAM_BOT_TOKEN)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
        ]
)

###################################################################################################################
class myBinance(Client):
    # client = Client(API_KEY, SECRET_KEY)
    # Client 클래스가 위와 같이 선언하므로, myBinance도 동일하게 해야 함.
    def __init__(self, api_key, secret_key) -> None:
        super().__init__(api_key, secret_key)

        # 인스턴스 최초 생성 시, 현재 밸런스와 전체 티커 정보를 얻어온다.
        self.current_spot_balance_df = self.update_current_spot_balance()
        self.symbol_info_df = self.update_all_ticker_info()


        self.target_symbols = ['BTC', 'ETH', 'BNB']
        self.usd_symbols = ['USDT', 'USDC', 'BUSD']

    
    def get_latest_klines_data(self, symbol, interval, latest_n=10):
        # api 호출 결과 데이터에 사용할 컬럼명.
        kline_colnames = [
            'open_time',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'close_time',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume',
            'ignore'
        ]
        # 숫자 형식으로 변환할 컬럼들.
        to_numeric_columns = [
            'open',
            'high',
            'low',
            'close',
            'volume',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume',
            'ignore'
        ]

        res = self.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE)
        kline_df = pd.DataFrame(res[-latest_n:], columns=kline_colnames)
        kline_df['open_time'] = pd.to_datetime(kline_df['open_time'], unit='ms')
        kline_df['close_time'] = pd.to_datetime(kline_df['close_time'], unit='ms')

        for col in to_numeric_columns:
            kline_df[col] = pd.to_numeric(kline_df[col], downcast='float')

        kline_df.set_index('open_time', drop=False, inplace=True)

        return kline_df.copy()

    def update_all_ticker_info(self):

        # 거래소 전체 토큰 정보 불러오기.
        all_info = self.get_exchange_info()
        rows = []
        
        # 필요한 정보(티커, 상태, base, quote)만 데이터프레임으로 저장.
        for info in all_info['symbols']:
            row = [info['symbol'], info['status'], info['baseAsset'], info['quoteAsset']]
            rows.append(row)
        all_symbol_info_df = pd.DataFrame(rows, columns=['symbol', 'status', 'base_asset', 'quote_asset'])
        
        return all_symbol_info_df.copy()

    def get_current_spot_balance(self):
        return self.current_spot_balance_df

    def update_current_spot_balance(self, nonzero_only=True):
        """현재 계정의 잔고 정보 가져오기(업데이트).
        Args:
            nonzero_only : True인 경우 잔고가 있는 토큰들의 정보만 출력.
        """

        # 계정정보 불러오기.
        account_info = self.get_account() # api 호출.

        # 계정 정보에서 잔고 정보만 저장.
        balance_df = pd.DataFrame(account_info["balances"])
        # 숫자 형태로 전환하고, free, locked를 합한 total 컬럼 계산.
        balance_df['free'] = pd.to_numeric(balance_df['free'])
        balance_df['locked'] = pd.to_numeric(balance_df['locked'])
        balance_df["total"] = balance_df["free"] + balance_df["locked"]

        # 0 초과 잔고만 저장.
        if nonzero_only:
            balance_df = balance_df[balance_df['total'] > 0]
        
        # total 컬럼 기준 내림차순 정렬 및 인덱스 재설정.
        # total 컬럼 값은 평가액이 아니고 해당 토큰의 amount이므로, 달러환산 가치와 다름.
        balance_df.sort_values(by='total', inplace=True, ascending=False)
        balance_df.reset_index(inplace=True, drop=True)
        return balance_df.copy()




def slope_sign(s):
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0

def strategy_signal_ma_cross(v1, v2, slope_check_n):
    ######################################################################
    # 크로스 여부 체크 로직.
    signal = [0]
    for i in range(len(v1)):
        if i == 0:
            continue
        
        # 현재 값이나 예전 값이 같은 경우. 무효
        if (v1[i] == v2[i]) or (v1[i-1] == v2[i-1]):
            signal.append(0)
            continue

        # v1 값이 v2 값을 하향돌파 하는 경우. 데드크로스.
        if (v1[i-1] > v2[i-1]) and (v1[i] < v2[i]):
            signal.append(-1)
        # v1 값이 v2 값을 상향돌파 하는 경우. 골든크로스.
        elif (v1[i-1] < v2[i-1]) and (v1[i] > v2[i]):
            signal.append(1)
        # 그 외의 경우 무효.
        else:
            signal.append(0)
    
    ######################################################################
    # 기울기 체크 로직. dataframe으로 계산 시 속도 느릴 수 있음. 추후 개선 필요.
    df = pd.DataFrame()
    df['v1'] = v1
    df['v2'] = v2
    df['signal_cross'] = signal
    df['v1_diff'] = df['v1'].diff(1).fillna(0)
    df['v2_diff'] = df['v2'].diff(1).fillna(0)

    all_check = []
    for r_df in df[['v1_diff', 'v2_diff']].rolling(slope_check_n):
        if r_df.shape[0] < slope_check_n:
            all_check.append(0)
            continue
        if np.all(r_df > 0):
            all_check.append(1)
        elif np.all(r_df < 0):
            all_check.append(-1)
        else:
            all_check.append(0)
    df['signal_all_slope'] = all_check
    df['signal_total'] = df['signal_cross'] * (df['signal_cross'] * df['signal_all_slope'])
    
    return df




###################################################################################################################
# 포지션 정보를 담고 있는 dataframe.
position_df = pd.DataFrame(columns=[
    'symbol',
    'open_time',
    'open_quote_asset_price',
    'position',
    'direction',
    'close_quote_asset_price',
    'close_time',
    'info',
    'ratio',
])
# 포지션 정보를 담고 있는 list. 각 포지션 정보는 dict.
position_list = []

bn = myBinance(API_KEY, SECRET_KEY)

position_duration = 3
latest_n = 20
target_symbol = 'BTCUSDT'
iter_i = 0

logging.info("< Trading start >")
while True:
    iter_i += 1
    time.sleep(10)

    try:
        # 봉차트 데이터 가져오기.
        df = bn.get_latest_klines_data(symbol=target_symbol, interval=Client.KLINE_INTERVAL_1MINUTE, latest_n=latest_n)
    except Exception as e:
        logging.error(e)
        continue
    
    # 이동평균 계산
    ma_periods = [3, 8]
    for p in ma_periods:
        new_colname = f"ma_{p}"
        df[new_colname] = df['close'].rolling(p).mean()

    # ma cross 전략 시그널 산출
    signal_df = strategy_signal_ma_cross(df['ma_3'], df['ma_8'], 3)
    cross_signals = signal_df['signal_cross'].tolist()
    total_signals = signal_df['signal_total'].tolist()

    # cross_signals[-1] = 1 # 임시 테스트용
    # 가장 최근의 시그널로 포지션 진입여부 판단.
    last_signal = cross_signals[-1]
    
    
    logging.info(f"iteration : {iter_i} / last 5 cross_signals : {cross_signals[-5:]}")
    
    
    # 크로스가 발생한 경우.
    if last_signal != 0:
        # 현재시간 저장.
        cur_time = datetime.datetime.now(datetime.timezone.utc)
        # 포지션 진입여부 저장.
        enter_position = True

        logging.info(f"  > Signal 발생 : {last_signal}")

        # (이전에 포지션 설정이 있었던 경우) 가장 최근 포지션 진입 시간 가져와서, 현재 시간과 비교하기.
        if len(position_list) > 0:
            # 가장 최근에 설정된 포지션만 저장.
            last_position = position_list[-1]
            # 해당 포지션 진입시간 저장.
            last_position_open_time = last_position['open_time']
            logging.info(f"  > 최근 포지션 open time : {last_position_open_time} / current : {cur_time}")

            # 현재시간 포함 최근 N분 이내 포지션 설정 있는 경우 스킵.
            # 시간 데이터를 분 단위까지로 저장.
            last_time_in_mins = last_position_open_time.replace(second=0, microsecond=0)
            cur_time_in_mins = cur_time.replace(second=0, microsecond=0)
            min_diff = cur_time_in_mins - last_time_in_mins
            skip_min = datetime.timedelta(minutes=5)
            if min_diff <= skip_min:
                logging.info(f"  > 최근 {skip_min} 분 내 진입한 포지션 있음. (차이 : {min_diff})")
                enter_position = False
        
        # 포지션 진입 조건을 만족하는 경우. 
        # 포지션 설정 히스토리가 없거나, 최근 설정이 없었던 경우. 포지션 진입.
        if enter_position:
            # 현재시간 저장.
            cur_time = datetime.datetime.now(datetime.timezone.utc)

            # 포지션 설정 정보 딕셔너리 저장.
            new_position = {
                'symbol':target_symbol,
                'open_time':cur_time,
                'open_quote_asset_price':df.loc[df.index[-1], 'close'],
                'position':'open',
                'direction':'long' if last_signal == 1 else 'short',
                'close_quote_asset_price':df.loc[df.index[-1], 'close'],
                'close_time':cur_time,
                'info':total_signals[-1],
            }
            new_position['ratio'] = new_position['close_quote_asset_price'] / new_position['open_quote_asset_price']
            # 리스트에 추가
            position_list.append(new_position)

            # dataframe에 추가.
            position_df = pd.concat([position_df, pd.DataFrame(new_position, index=[0])], ignore_index=True)

            logging.info(f"  > 신규 포지션 진입 완료.")
            logging.info(f"  > 현재 포지션 정보\n{position_df}")

            telegram_text = f"➤ 신규 포지션 진입 완료.({cur_time.strftime('%Y-%m-%d %H:%M:%S')} utc)\n"
            telegram_text += "\n<b>[ 현재 포지션 정보 ]</b>\n"
            for idx in position_df.index:
                t = position_df.loc[idx, 'open_time'].strftime('%m/%d %H:%M')
                d = position_df.loc[idx, 'direction']
                p = position_df.loc[idx, 'position']
                op = round(position_df.loc[idx, 'open_quote_asset_price'], 2)
                cp = round(position_df.loc[idx, 'close_quote_asset_price'], 2)
                r = round((cp/op-1)*100, 3)
                info = position_df.loc[idx, 'info']

                position_text = f"{t}|{d}|{p}|{op}|{cp}|{r}%|{info}"

                if idx == position_df.index[-1]:
                    telegram_text += "<b>" + position_text + "</b>\n"
                else:
                    telegram_text += position_text + "\n"
            tele_bot.sendMessage(chat_id=TELEGRAM_CHAT_ID, text=telegram_text, parse_mode='HTML')
    
    # 최근가격으로 업데이트.(포지션 있는 경우에만.)
    if position_df.shape[0] > 0:

        try:
            # 최근 거래정보 가져오기.
            recent_trade = bn.get_recent_trades(symbol='BTCUSDT', limit=1)[0]
        except Exception as e:
            logging.error(e)
            continue
        
        # 최근 거래의 시간정보.
        trade_time = pd.to_datetime(recent_trade['time'], unit='ms', utc=True)

        # 현재 유지중인(open) 포지션에 대해서만 현재가격으로 업데이트 한다.
        open_position_index = position_df[position_df['position'] == 'open'].index
        for idx in open_position_index:
            # 최근가격, 최근시간으로 업데이트.
            position_df.loc[idx, 'close_quote_asset_price'] = float(recent_trade['price'])
            position_df.loc[idx, 'close_time'] = trade_time
            # 포지션 진입 시간에서 일정 시간이 지난 경우 포지션 종료.
            if trade_time - position_df.loc[idx, 'open_time'] >= datetime.timedelta(minutes=position_duration):
                position_df.loc[idx, 'position'] = 'closed'
                logging.info(f"  > position index({idx}) 포지션 종료. 포지션 지속기간 초과.")

        # 수익률 계산
        position_df['ratio'] = position_df['close_quote_asset_price'] / position_df['open_quote_asset_price']
        position_df.to_csv('position.csv')




# fig = mpf.figure(style='charles',figsize=(7,8))
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(3,1,3)   
    
# def animate(i):
    # # 차트 초기화
    # ax1.clear()
    # ax2.clear()

    # # 차트 생성시 계산되는 값들 저장할 dict.
    # cv = {}

    # # 봉차트 생성 및 추가 계산 처리.
    # mpf.plot(df, ax=ax1, volume=ax2, type='candle', mav=(5,10,25), return_calculated_values=cv)

# ani = FuncAnimation(fig, animate, interval=1000)
# mpf.show()



# print(res)
# print("type :", type(res))
# print("length :", len(res))

# if type(res) == type({}):
#     for k, v in res.items():
#         print(k)
# elif type(res) == type([]):
#     for k, v in res[0].items():
#         print(k, ":", v)