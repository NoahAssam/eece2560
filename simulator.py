import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class StockTrader:
    """
    A stock trading system that implements greedy algorithms and technical analysis
    for making trading decisions.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.holdings: Dict[str, int] = {}  # {stock_symbol: quantity}
        self.trades: List[dict] = []
        self.profit_loss = 0.0

    def fetch_stock_data(self, symbol: str, duration: str = '1mo') -> pd.DataFrame:
        """
        Fetches historical stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            duration: Time period for data (e.g., '1mo', '3mo', '1y')

        Returns:
            DataFrame with stock price history
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=duration)
            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")

    def calculate_technical_indicators(self,
                                       price_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculates technical indicators for trading decisions.

        Args:
            price_data: DataFrame with price history

        Returns:
            Tuple of (short_term_ma, long_term_ma, volatility)
        """
        short_term_ma = price_data['Close'].rolling(window=20).mean()
        long_term_ma = price_data['Close'].rolling(window=50).mean()
        volatility = price_data['Close'].rolling(window=20).std()

        return short_term_ma, long_term_ma, volatility

    def analyze_market_trend(self,
                             current_price: float,
                             short_ma: float,
                             long_ma: float,
                             volatility: float) -> str:
        """
        Analyzes market trend using technical indicators.

        Returns:
            'uptrend', 'downtrend', or 'neutral'
        """
        if short_ma > long_ma and current_price > short_ma:
            return 'uptrend'
        elif short_ma < long_ma and current_price < short_ma:
            return 'downtrend'
        return 'neutral'

    def generate_trade_signal(self,
                              symbol: str,
                              current_price: float,
                              trend: str,
                              volatility: float) -> str:
        """
        Generates trading signal based on market analysis.

        Returns:
            'buy', 'sell', or 'hold'
        """
        # Risk management: Don't trade if volatility is too high
        if volatility > current_price * 0.05:  # 5% volatility threshold
            return 'hold'

        if trend == 'uptrend' and symbol not in self.holdings:
            if self.balance >= current_price:
                return 'buy'
        elif trend == 'downtrend' and symbol in self.holdings:
            return 'sell'

        return 'hold'

    def execute_trade(self,
                      symbol: str,
                      action: str,
                      price: float,
                      quantity: int = 1) -> bool:
        """
        Executes a trade based on the given parameters.

        Returns:
            bool indicating if trade was successful
        """
        timestamp = datetime.now()

        if action == 'buy':
            total_cost = price * quantity
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'total': total_cost
                })
                return True

        elif action == 'sell':
            if symbol in self.holdings and self.holdings[symbol] >= quantity:
                total_value = price * quantity
                self.balance += total_value
                self.holdings[symbol] -= quantity
                if self.holdings[symbol] == 0:
                    del self.holdings[symbol]
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'total': total_value
                })
                return True

        return False

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Calculates current value of a position."""
        return self.holdings.get(symbol, 0) * current_price

    def calculate_portfolio_value(self,
                                  price_data: Dict[str, float]) -> float:
        """
        Calculates total portfolio value including cash balance.
        """
        total_value = self.balance
        for symbol, quantity in self.holdings.items():
            if symbol in price_data:
                total_value += quantity * price_data[symbol]
        return total_value

    def generate_performance_report(self) -> Dict:
        """
        Generates a comprehensive performance report.
        """
        total_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t['action'] == 'buy'])
        sell_trades = len([t for t in self.trades if t['action'] == 'sell'])

        report = {
            'current_balance': round(self.balance, 2),
            'current_holdings': self.holdings,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'profit_loss': round(self.profit_loss, 2),
            'trade_history': self.trades[-5:]  # Last 5 trades
        }
        return report

    def visualize_trades(self,
                         symbol: str,
                         price_data: pd.DataFrame) -> None:
        """
        Creates visualization of trading activity.
        """
        plt.figure(figsize=(12, 6))

        # Plot price data
        plt.plot(price_data.index, price_data['Close'], label='Price', color='blue')

        # Plot buy points
        buy_trades = [t for t in self.trades if t['action'] == 'buy' and t['symbol'] == symbol]
        if buy_trades:
            buy_x = [t['timestamp'] for t in buy_trades]
            buy_y = [t['price'] for t in buy_trades]
            plt.scatter(buy_x, buy_y, color='green', marker='^', label='Buy')

        # Plot sell points
        sell_trades = [t for t in self.trades if t['action'] == 'sell' and t['symbol'] == symbol]
        if sell_trades:
            sell_x = [t['timestamp'] for t in sell_trades]
            sell_y = [t['price'] for t in sell_trades]
            plt.scatter(sell_x, sell_y, color='red', marker='v', label='Sell')

        plt.title(f'Trading Activity - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage class
class TradingSimulation:
    """
    Handles the simulation of trading strategies.
    """

    def __init__(self, trader: StockTrader):
        self.trader = trader

    def run_simulation(self,
                       symbol: str,
                       duration: str = '3mo',
                       trade_interval: str = 'daily') -> Dict:
        """
        Runs a trading simulation for the specified period.
        """
        # Fetch historical data
        price_data = self.trader.fetch_stock_data(symbol, duration)

        # Run simulation for each interval
        for i in range(len(price_data) - 1):
            current_data = price_data.iloc[:i + 1]
            current_price = current_data['Close'].iloc[-1]

            # Calculate indicators
            short_ma, long_ma, volatility = self.trader.calculate_technical_indicators(current_data)

            # Generate trading signal
            trend = self.trader.analyze_market_trend(
                current_price,
                short_ma.iloc[-1],
                long_ma.iloc[-1],
                volatility.iloc[-1]
            )

            signal = self.trader.generate_trade_signal(
                symbol,
                current_price,
                trend,
                volatility.iloc[-1]
            )

            # Execute trade if signal is not 'hold'
            if signal != 'hold':
                self.trader.execute_trade(symbol, signal, current_price)

        # Generate final report
        return self.trader.generate_performance_report()


if __name__ == "__main__":
    # Create trader instance
    trader = StockTrader(initial_balance=10000.0)

    # Create simulation instance
    simulation = TradingSimulation(trader)

    # Run simulation for AAPL
    results = simulation.run_simulation('AAPL', duration='3mo')

    # Print results
    print("\nSimulation Results:")
    print("-" * 50)
    print(f"Final Balance: ${results['current_balance']}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profit/Loss: ${results['profit_loss']}")
    print("\nCurrent Holdings:")
    for symbol, quantity in results['current_holdings'].items():
        print(f"{symbol}: {quantity} shares")
    print("\nRecent Trades:")
    for trade in results['trade_history']:
        print(
            f"{trade['timestamp']}: {trade['action'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']}")