
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from stock_trader import StockTrader, TradingSimulation


class TestStockTrader(unittest.TestCase):
    """Test suite for StockTrader class"""

    def setUp(self):
        """Set up test environment"""
        self.trader = StockTrader(initial_balance=10000.0)

        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100)
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(100, 120, 100),
            'Low': np.random.uniform(80, 100, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

    def test_initialization(self):
        """Test trader initialization"""
        self.assertEqual(self.trader.balance, 10000.0)
        self.assertEqual(len(self.trader.holdings), 0)
        self.assertEqual(len(self.trader.trades), 0)
        self.assertEqual(self.trader.profit_loss, 0.0)

    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        short_ma, long_ma, volatility = self.trader.calculate_technical_indicators(self.sample_data)

        self.assertEqual(len(short_ma), len(self.sample_data))
        self.assertEqual(len(long_ma), len(self.sample_data))
        self.assertEqual(len(volatility), len(self.sample_data))

        # First values should be NaN due to rolling window
        self.assertTrue(pd.isna(short_ma.iloc[0]))
        self.assertTrue(pd.isna(long_ma.iloc[0]))
        self.assertTrue(pd.isna(volatility.iloc[0]))

    def test_market_trend_analysis(self):
        """Test market trend analysis"""
        # Test uptrend
        trend = self.trader.analyze_market_trend(
            current_price=105,
            short_ma=100,
            long_ma=95,
            volatility=2
        )
        self.assertEqual(trend, 'uptrend')

        # Test downtrend
        trend = self.trader.analyze_market_trend(
            current_price=90,
            short_ma=95,
            long_ma=100,
            volatility=2
        )
        self.assertEqual(trend, 'downtrend')

    def test_trade_signal_generation(self):
        """Test trade signal generation"""
        # Test buy signal
        signal = self.trader.generate_trade_signal(
            symbol='AAPL',
            current_price=100,
            trend='uptrend',
            volatility=2
        )
        self.assertEqual(signal, 'buy')

        # Test high volatility hold
        signal = self.trader.generate_trade_signal(
            symbol='AAPL',
            current_price=100,
            trend='uptrend',
            volatility=10
        )
        self.assertEqual(signal, 'hold')

    def test_trade_execution(self):
        """Test trade execution"""
        # Test successful buy
        success = self.trader.execute_trade('AAPL', 'buy', 100, 1)
        self.assertTrue(success)
        self.assertEqual(self.trader.balance, 9900.0)
        self.assertEqual(self.trader.holdings['AAPL'], 1)

        # Test successful sell
        success = self.trader.execute_trade('AAPL', 'sell', 110, 1)
        self.assertTrue(success)
        self.assertEqual(self.trader.balance, 10010.0)
        self.assertEqual(len(self.trader.holdings), 0)

    def test_invalid_trades(self):
        """Test invalid trade scenarios"""
        # Test insufficient balance
        success = self.trader.execute_trade('AAPL', 'buy', 20000, 1)
        self.assertFalse(success)

        # Test selling without holdings
        success = self.trader.execute_trade('AAPL', 'sell', 100, 1)
        self.assertFalse(success)

    def test_portfolio_calculations(self):
        """Test portfolio value calculations"""
        self.trader.execute_trade('AAPL', 'buy', 100, 1)
        position_value = self.trader.get_position_value('AAPL', 110)
        self.assertEqual(position_value, 110)

        portfolio_value = self.trader.calculate_portfolio_value({'AAPL': 110})
        self.assertEqual(portfolio_value, 9900 + 110)

    def test_performance_report(self):
        """Test performance report generation"""
        self.trader.execute_trade('AAPL', 'buy', 100, 1)
        self.trader.execute_trade('AAPL', 'sell', 110, 1)

        report = self.trader.generate_performance_report()

        self.assertIn('current_balance', report)
        self.assertIn('current_holdings', report)
        self.assertIn('total_',report)