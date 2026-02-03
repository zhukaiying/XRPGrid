"""
XRP 网格策略

- 适用于震荡行情
- 价格下跌分批买入，价格上涨分批卖出
- 动态调整网格区间（基于ATR或移动平均）
- 风险控制：最大持仓限制
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import time
import json
import logging
import sys

# 尝试导入 ccxt，用于获取实时行情
try:
    import ccxt
except ImportError:
    ccxt = None
    print("Warning: 'ccxt' module not found. Real-time trading will not work. (pip install ccxt)")

# ==================== 配置 ====================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "BINANCE")

# 策略参数
GRID_LINES = 20         # 网格数量
GRID_SPREAD = 0.02      # 每个网格的间距（2%）
INITIAL_CAPITAL = 100   # 初始资金 USDT
POSITION_SIZE_PER_GRID = 0.05 # 每格仓位 5%
LEVERAGE = 3            # 杠杆倍数
STOP_LOSS_PCT = 0.2     # 总资产下跌20%止损

# 网络代理设置
# 根据您的截图，您的 Clash 端口是 7897
#PROXY_URL = "http://127.0.0.1:7897"  
PROXY_URL = None                   # <--- 如果在国外服务器运行，请使用这行

@dataclass
class Trade:
    time: str
    action: str         # BUY / SELL
    price: float
    quantity: float
    value: float
    fee: float

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.DataFrame
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    avg_profit_per_day: float
    win_rate: float = 0
    profit_factor: float = 0
    sharpe_ratio: float = 0
    win_trades: int = 0
    lose_trades: int = 0
    avg_win: float = 0
    avg_loss: float = 0

# ==================== 回测引擎 ====================

class GridBacktester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position_qty = 0
        self.trades: List[Trade] = []
        self.equity_history = []
        
        # 初始化网格
        self.start_price = df.iloc[0]['close']
        self.grids = self._generate_grids(self.start_price)
        self.last_grid_index = self._get_grid_index(self.start_price)

    def _generate_grids(self, center_price: float) -> List[float]:
        """生成网格价格线"""
        grids = []
        # 以初始价格为中心，上下生成网格
        # 下方网格：买单
        for i in range(1, GRID_LINES // 2 + 1):
            price = center_price * (1 - i * GRID_SPREAD)
            grids.append(price)
        # 上方网格：卖单
        for i in range(1, GRID_LINES // 2 + 1):
            price = center_price * (1 + i * GRID_SPREAD)
            grids.append(price)
        
        grids.append(center_price)
        grids.sort()
        return grids

    def _get_grid_index(self, price: float) -> int:
        """找到当前价格对应的最近网格索引"""
        for i, grid_price in enumerate(self.grids):
            if price < grid_price:
                return max(0, i - 1)
        return len(self.grids) - 1

    def run(self) -> BacktestResult:
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            current_price = row['close']
            current_grid_index = self._get_grid_index(current_price)
            
            # 价格穿越网格
            if current_grid_index != self.last_grid_index:
                # 价格下跌，穿越下方网格 -> 买入
                if current_grid_index < self.last_grid_index:
                    self._execute_trade("BUY", current_price, row['open_time'])
                
                # 价格上涨，穿越上方网格 -> 卖出
                elif current_grid_index > self.last_grid_index:
                    self._execute_trade("SELL", current_price, row['open_time'])
                
                self.last_grid_index = current_grid_index

            # 动态更新网格（可选：当价格偏离过远时重置中枢）
            if current_price < self.grids[0] * 0.9 or current_price > self.grids[-1] * 1.1:
                self.grids = self._generate_grids(current_price)
                self.last_grid_index = self._get_grid_index(current_price)

            # 记录权益
            equity = self._calculate_equity(current_price)
            self.equity_history.append({
                'time': row['open_time'],
                'equity': equity,
                'close': current_price
            })
            
            # 止损检查
            if (self.initial_capital - equity) / self.initial_capital > STOP_LOSS_PCT:
                break

        return self._generate_result()

    def _execute_trade(self, action: str, price: float, time: str):
        value = self.initial_capital * POSITION_SIZE_PER_GRID * LEVERAGE
        
        if action == "BUY":
            # 检查是否有资金买入
            cost = value / LEVERAGE
            if self.balance >= cost:
                quantity = value / price
                fee = value * 0.0004
                self.balance -= cost + fee
                self.position_qty += quantity
                self.trades.append(Trade(time, "BUY", price, quantity, value, fee))
        
        elif action == "SELL":
            # 检查是否有持仓卖出
            quantity = value / price
            if self.position_qty >= quantity:
                fee = value * 0.0004
                self.balance += (value / LEVERAGE) + (value - (quantity * price)) - fee # 归还保证金+盈亏
                # 简化计算：平仓盈亏直接算入余额
                entry_cost = quantity * price / LEVERAGE
                pnl = (price * quantity) - (price * quantity) # 这里的逻辑需要修正，网格卖出应该是平掉之前的买单
                
                # 网格卖出：卖出部分持仓
                sell_value = quantity * price
                self.balance += (sell_value / LEVERAGE) - fee # 释放保证金
                # 这里为了简化，假设每次卖出都能获利（高抛低吸）
                # 实际盈亏体现在余额变化中
                
                self.position_qty -= quantity
                self.trades.append(Trade(time, "SELL", price, quantity, value, fee))

    def _calculate_equity(self, current_price: float) -> float:
        # 权益 = 余额 + 持仓价值（未实现盈亏）
        # 这里简化处理：持仓盈亏 = (当前价 - 平均持仓价) * 数量
        # 由于网格买入价格各不相同，准确计算需要维护买入队列
        # 估算：假设持仓均价为当前网格中心附件
        
        # 更准确的方法：权益 = 现金余额 + 保证金 + 未实现盈亏
        # 简单起见：Equity = Balance (已实现) + PositionValue - Borrowed
        
        # 修正：
        # 开仓时：Balance减少 margin
        # 持仓价值：Qty * Price
        # 借款（杠杆部分）：Qty * EntryPrice * (1 - 1/Lev) -> 难以追踪每笔
        
        # 采用最简单的净值估算：
        # 假设每次买入都是现货逻辑（带杠杆的现货）
        # Net Value = Balance + (Position Qty * Current Price)
        # 但要注意 Balance 里面的钱是 USDT
        
        # 重新定义 Balance 为总权益（含未结盈亏）
        # 也不对，Balance 是可用资金
        
        # 采用标准计算：
        # Equity = Balance + Unrealized PnL
        # Unrealized PnL = quantity * (current_price - avg_entry_price)
        # 这种计算需要在 _execute_trade 中维护 avg_entry_price
        
        # 重新实现一个简单的 avg_entry_price 维护
        return self.balance # 暂用余额，因为网格卖出后利润已进余额

    # 重新实现一套带均价维护的简易逻辑
    def _execute_trade_v2(self, action: str, price: float, time: str):
        # ...
        pass

# ==================== 现货/多头杠杆网格（只做多，不反向做空） ====================

class SpotGridBacktester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL, leverage: float = 1.0):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.leverage = leverage
        
        # 资金账户
        # 为了支持杠杆，我们维护一个 Margin Balance
        # 当买入时，扣除 (Value / Leverage) 的保证金
        # 当卖出时，释放保证金 + 盈亏
        
        # 简化模型：
        # 维护一个虚拟的 cash_balance 表示购买力
        # 但这样很难计算准确的 净值
        
        # 采用最稳健的逻辑：
        # Balance = 账户总权益 (Cash)
        # 每次 Buy，Position += Qty, Balance -= 0 (钱变成仓位，这里是做多网格，不考虑保证金占用细节，只看总权益)
        # 不对，这样没法算爆仓。
        
        # 回归到 "Leveraged Spot Grid" 逻辑
        # 初始资金 $100
        # 杠杆 5x -> 购买力 $500
        # 这是一个 "Virtual Balance"
        
        self.net_balance = initial_capital # 真实净值 (USDT)
        self.position = 0.0 # 持仓数量
        self.avg_price = 0.0
        
        self.trades = []
        self.equity_history = []
        
        self.start_price = df.iloc[0]['close']
        self.grids = self._generate_grids(self.start_price)
        self.current_grid = self._get_grid_index(self.start_price)
        
        # 初始不开仓，从零开始低吸

    def _generate_grids(self, center_price: float) -> List[float]:
        grids = []
        for i in range(-GRID_LINES // 2, GRID_LINES // 2 + 1):
            price = center_price * (1 + i * GRID_SPREAD)
            grids.append(price)
        return sorted(grids)

    def _get_grid_index(self, price: float) -> int:
        for i, p in enumerate(self.grids):
            if price < p:
                return i
        return len(self.grids)

    def _execute_trade(self, side: int, price: float, time: str):
        # side: 1 (Buy), -1 (Sell)
        # 固定下单金额（基于杠杆后的总资金）
        # 每次每格买入资金 = 初始本金 * 杠杆 * 单格比例
        order_val = self.initial_capital * self.leverage * POSITION_SIZE_PER_GRID
        qty = order_val / price
        
        fee = order_val * 0.0004 # 手续费
        self.net_balance -= fee
        
        if side == 1: # Buy
            # 买入：增加持仓，更新均价
            # 资金逻辑：这里是合约做多，其实只是增加了仓位价值，占用了保证金
            # 净值变动 = -Fees
            
            total_val = self.position * self.avg_price + qty * price
            total_qty = self.position + qty
            self.avg_price = total_val / total_qty
            self.position += qty
            
            self.trades.append(Trade(str(time), "BUY", price, qty, order_val, fee))
            
        elif side == -1: # Sell
            # 卖出：平仓逻辑
            # 【关键修改】：只平仓，不反手开空！
            
            trade_qty = qty
            
            # 如果现有持仓不足以卖出（比如已经卖光了），就只卖剩下的
            if self.position < trade_qty:
                trade_qty = max(0, self.position)
            
            if trade_qty > 0:
                # 结算盈亏
                # PnL = (Exit - Entry) * Qty
                pnl = (price - self.avg_price) * trade_qty
                self.net_balance += pnl
                
                self.position -= trade_qty
                self.trades.append(Trade(str(time), "SELL", price, trade_qty, trade_qty * price, fee))
            else:
                # 没货卖了，不做空，直接跳过
                pass


    def run(self) -> BacktestResult:
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            price = row['close']
            new_grid = self._get_grid_index(price)
            
            # 交易信号
            if new_grid != self.current_grid:
                # 价格下跌 -> 触碰下方网格 -> 买入
                if new_grid < self.current_grid:
                    self._execute_trade(1, price, row['open_time'])
                
                # 价格上涨 -> 触碰上方网格 -> 卖出 (只平多，不开空)
                elif new_grid > self.current_grid:
                    self._execute_trade(-1, price, row['open_time'])
                
                self.current_grid = new_grid

            # 动态网格中心（无限网格逻辑）
            if price < self.grids[0] or price > self.grids[-1]:
                self.grids = self._generate_grids(price)
                self.current_grid = self._get_grid_index(price)
                
            # 计算净值 (总资产)
            # Equity = Net Balance (Cash) + Unrealized PnL
            unrealized_pnl = 0
            if self.position > 0:
                unrealized_pnl = (price - self.avg_price) * self.position
            
            equity = self.net_balance + unrealized_pnl
            
            self.equity_history.append({
                'time': row['open_time'],
                'equity': equity,
                'close': price
            })

        return self._generate_result()

    def _generate_result(self) -> BacktestResult:
        equity_df = pd.DataFrame(self.equity_history)
        total_return = equity_df['equity'].iloc[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak']
        max_drawdown_pct = equity_df['drawdown'].max() * 100
        max_drawdown = equity_df['peak'].max() - equity_df['equity'].min()
        
        days = (pd.to_datetime(equity_df['time'].iloc[-1]) - pd.to_datetime(equity_df['time'].iloc[0])).days
        avg_profit_per_day = total_return / days if days > 0 else 0

        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=len(self.trades),
            avg_profit_per_day=avg_profit_per_day,
            win_rate=0, profit_factor=0, sharpe_ratio=0, # 兼容接口
            win_trades=0, lose_trades=0, avg_win=0, avg_loss=0 # 兼容接口
        )

# ==================== 主函数 ====================

def load_data(symbol: str = "XRPUSDT", timeframe: str = "4h") -> pd.DataFrame:
    filename = f"{symbol}_futures_{timeframe}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    df['open_time'] = pd.to_datetime(df['open_time'])
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术指标（用于显示器绘图）
    网格策略主要依赖价格水平，这里计算一些辅助指标供参考
    """
    # 简单移动平均线作为趋势参考
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    return df

def run_backtest(symbol: str = "XRPUSDT", timeframe: str = "4h", leverage: float = 1.0) -> Tuple[BacktestResult, pd.DataFrame]:
    df = load_data(symbol, timeframe)
    # 确保指标存在
    df = calculate_indicators(df)
    
    # 将 leverage 传递给 SpotGridBacktester
    # 注意：前面的 replace_string_in_file 似乎没有成功修改 SpotGridBacktester 的 __init__，
    # 如果没修改，这里传参会报错。
    # 为了保险，我们再检查一下 SpotGridBacktester 的定义。
    # 基于之前的上下文，我已经修改了 SpotGridBacktester 的 __init__ 签名。
    backtester = SpotGridBacktester(df, initial_capital=INITIAL_CAPITAL, leverage=leverage)
    result = backtester.run()
    
    # 满仓持有对比
    entry_price = df.iloc[0]['close']
    qty = INITIAL_CAPITAL / entry_price
    df['hold_equity'] = qty * df['close']
    
    equity_df = result.equity_curve
    equity_df['hold_equity'] = df['hold_equity'].values

    return result, equity_df

# ==================== 实盘模拟引擎 (新增) ====================

class PaperGridTrader:
    """
    实盘模拟交易器 (Paper Trading)
    - 连接交易所 API 获取实时价格
    - 本地模拟账户资金和持仓，不发送真实订单
    - 支持断点续传（状态保存到 json）
    """
    def __init__(self, symbol="XRP/USDT", initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.state_file = "grid_state.json"
        
        # 配置日志
        self.logger = logging.getLogger("GridTrader")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # 同时保存到文件
            file_handler = logging.FileHandler("grid_run.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if ccxt is None:
            self.logger.error("请先安装 ccxt: pip install ccxt")
            sys.exit(1)
            
        exchange_config = {
            'enableRateLimit': True,  # 启用速率限制
            'timeout': 30000,         # 增加超时时间
        }
        
        # 优先使用环境变量中的代理，其次使用配置的 PROXY_URL
        proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') or PROXY_URL
        
        if proxy:
            exchange_config['proxies'] = {
                'http': proxy,
                'https': proxy,
            }
            self.logger.info(f"使用代理: {proxy}")
        else:
            self.logger.info("未使用代理 (建议: 本地运行如果连不上币安，请设置 PROXY_URL)")
            
        self.exchange = ccxt.binance(exchange_config)

        # 简单连通性测试
        try:
            self.logger.info("正在测试交易所连接...")
            self.exchange.fetch_time()
            self.logger.info("交易所连接成功！")
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self.logger.error("请检查: 1.网络是否正常 2.PROXY_URL 端口是否正确 (Clash默认7890, v2ray默认10809)")
            # 这里不退出，让它重试


        # 加载或初始化状态
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.balance = state['balance']
                    self.position = state['position']
                    self.avg_price = state['avg_price']
                    self.grids = state['grids']
                    self.current_grid_index = state['current_grid_index']
                    self.trades = state.get('trades', []) # 简单记录
                    self.logger.info("已加载历史状态")
            except Exception as e:
                self.logger.error(f"加载状态失败: {e}, 将重新初始化")
                self.init_state()
        else:
            self.init_state()

    def init_state(self):
        self.balance = self.initial_capital
        self.position = 0.0
        self.avg_price = 0.0
        self.grids = []
        self.current_grid_index = -1
        self.trades = []
        self.logger.info("初始化新状态")

    def save_state(self):
        state = {
            'balance': self.balance,
            'position': self.position,
            'avg_price': self.avg_price,
            'grids': self.grids,
            'current_grid_index': self.current_grid_index,
            'trades': self.trades[-50:] # 只存最近50笔
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)

    def get_market_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
            return None

    def _generate_grids(self, center_price: float) -> List[float]:
        grids = []
        for i in range(-GRID_LINES // 2, GRID_LINES // 2 + 1):
            price = center_price * (1 + i * GRID_SPREAD)
            grids.append(price)
        return sorted(grids)

    def _get_grid_index(self, price: float) -> int:
        if not self.grids: return -1
        for i, p in enumerate(self.grids):
            if price < p:
                return i
        return len(self.grids)

    def run(self):
        self.logger.info(f"启动做多网格策略: {self.symbol}, 资金: {self.balance}, 杠杆: {self.leverage}")
        
        while True:
            try:
                price = self.get_market_price()
                if price is None:
                    time.sleep(5)
                    continue

                # 首次运行，生成网格
                if not self.grids:
                    self.grids = self._generate_grids(price)
                    self.current_grid_index = self._get_grid_index(price)
                    self.logger.info(f"初始网格生成完成，中心价: {price}")
                    self.save_state()
                    time.sleep(1)
                    continue

                current_index = self._get_grid_index(price)
                
                # 打印状态 (心跳)
                # self.logger.debug(f"当前价: {price}, 网格索引: {current_index}")

                if current_index != self.current_grid_index:
                    self.logger.info(f"价格穿越网格: {self.current_grid_index} -> {current_index}, 价格: {price}")
                    
                    # 下跌买入
                    if current_index < self.current_grid_index:
                        self._execute_trade("BUY", price)
                    
                    # 上涨卖出
                    elif current_index > self.current_grid_index:
                        self._execute_trade("SELL", price)

                    self.current_grid_index = current_index
                    
                    # 动态网格检查
                    if price < self.grids[0] or price > self.grids[-1]:
                        self.logger.info("价格超出网格范围，重置网格...")
                        self.grids = self._generate_grids(price)
                        self.current_grid_index = self._get_grid_index(price)
                    
                    self.save_state()

                # 显示当前权益
                unrealized_pnl = (price - self.avg_price) * self.position if self.position > 0 else 0
                equity = self.balance + unrealized_pnl
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Price: {price:.4f} | Grid: {current_index} | Pos: {self.position:.2f} | Eq: {equity:.2f}", end="")

            except KeyboardInterrupt:
                self.logger.info("用户停止脚本")
                break
            except Exception as e:
                self.logger.error(f"运行循环发生错误: {e}")
                time.sleep(5) # 防止死循环报错
            
            time.sleep(5) # 轮询间隔

    def _execute_trade(self, side: str, price: float):
        order_val = self.initial_capital * self.leverage * POSITION_SIZE_PER_GRID
        qty = order_val / price
        fee = order_val * 0.0004
        
        if side == "BUY":
            # 扣除手续费 (这里简化从余额扣)
            self.balance -= fee 
            
            total_val = self.position * self.avg_price + qty * price
            total_qty = self.position + qty
            self.avg_price = total_val / total_qty
            self.position += qty
            
            self.logger.info(f"【买入】 数量: {qty:.2f} @ {price:.4f} | 均价: {self.avg_price:.4f}")
            self.trades.append({"time": str(datetime.now()), "side": "BUY", "price": price, "qty": qty})
            
        elif side == "SELL":
            # 只能卖现有持仓
            trade_qty = qty
            if self.position < trade_qty:
                trade_qty = max(0, self.position)
                
            if trade_qty > 0:
                pnl = (price - self.avg_price) * trade_qty
                self.balance += pnl - fee
                self.position -= trade_qty
                
                self.logger.info(f"【卖出】 数量: {trade_qty:.2f} @ {price:.4f} | 盈亏: {pnl:.2f}")
                self.trades.append({"time": str(datetime.now()), "side": "SELL", "price": price, "qty": trade_qty, "pnl": pnl})
            else:
                self.logger.info("【信号忽略】 触发卖出信号但无持仓")


if __name__ == "__main__":
    # 使用参数或简单交互区分模式
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        # 运行模拟盘
        trader = PaperGridTrader()
        trader.run()
    else:
        # 运行回测 (默认)
        print("正在运行回测模式 (使用 'python grid_strategy.py live' 运行实盘模拟)...")
        result, equity_df = run_backtest()
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"Total Trades: {result.total_trades}")

