import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import alpaca_trade_api as tradeapi
import yfinance as yf
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionTradingBot:
    def __init__(self, api_key, secret_key, base_url, symbol='AAPL'):
        """
        Initialize the trading bot

        Args:
            api_key (str): Alpaca API key
            secret_key (str): Alpaca secret key
            base_url (str): Alpaca base URL (paper: https://paper-api.alpaca.markets)
            symbol (str): Stock symbol to trade
        """
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.symbol = symbol
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.lookback_days = 20  # Days of data to use for prediction (reduced for reliability)
        self.position_size = 100  # Number of shares to trade

        # Trading parameters
        self.buy_threshold = 0.02  # Buy if predicted return > 2%
        self.sell_threshold = -0.01  # Sell if predicted return < -1%

    def get_historical_data(self, days=60):
        """
        Fetch historical stock data from Yahoo Finance (free alternative)

        Args:
            days (int): Number of days of historical data to fetch

        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            # Try Alpaca first (for users with paid subscriptions)
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                bars = self.api.get_bars(
                    self.symbol,
                    '1Day',
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df

                # Reset index to get timestamp as column
                bars = bars.reset_index()
                bars['Date'] = bars['timestamp'].dt.strftime('%Y-%m-%d')

                # Rename columns to match expected format
                bars = bars.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })

                logger.info(f"Fetched {len(bars)} days of data for {self.symbol} from Alpaca")
                return bars[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            except Exception as alpaca_error:
                logger.warning(f"Alpaca data fetch failed: {alpaca_error}")
                logger.info("Falling back to Yahoo Finance...")

                # Fallback to Yahoo Finance
                end_date = datetime.now()
                start_date = end_date - timedelta(
                    days=days + 30)  # Get extra days to ensure enough data after weekends/holidays

                # Download data from Yahoo Finance
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )

                if df.empty:
                    logger.error(f"No data found for symbol {self.symbol}")
                    return None

                # Reset index and format data
                df = df.reset_index()
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

                # Rename columns to match expected format
                df = df.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                })

                # Select only the columns we need and get the most recent data
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(days)

                logger.info(f"Fetched {len(df)} days of data for {self.symbol} from Yahoo Finance")
                logger.info(f"Data range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
                return df

        except Exception as e:
            logger.error(f"Error fetching historical data from all sources: {e}")
            return None

    def create_features(self, df):
        """
        Create features for linear regression model

        Args:
            df (pd.DataFrame): Stock data

        Returns:
            pd.DataFrame: DataFrame with features
        """
        df = df.copy()

        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Price ratios
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2'] = df['Close'].pct_change(2)
        df['Price_Change_5'] = df['Close'].pct_change(5)

        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()

        # Target variable (next day return)
        df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1

        # Drop rows with NaN values
        df = df.dropna()

        return df

    def prepare_training_data(self, df):
        """
        Prepare data for model training

        Args:
            df (pd.DataFrame): DataFrame with features

        Returns:
            tuple: (X_scaled, y) training data
        """
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'High_Low_Ratio', 'Close_Open_Ratio',
            'Volume_Ratio', 'Price_Change', 'Price_Change_2', 'Price_Change_5', 'Volatility'
        ]

        X = df[feature_columns].values
        y = df['Next_Day_Return'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_model(self, df):
        """
        Train the linear regression model

        Args:
            df (pd.DataFrame): Training data
        """
        try:
            logger.info(f"Starting model training with {len(df)} days of data")

            # Create features
            df_features = self.create_features(df)

            logger.info(f"After feature creation: {len(df_features)} days of data")

            if len(df_features) < self.lookback_days:
                logger.warning(
                    f"Not enough data for training. Have {len(df_features)} days, need at least {self.lookback_days} days")
                # Try with available data if we have at least 10 days
                if len(df_features) >= 10:
                    logger.info(f"Proceeding with {len(df_features)} days of data")
                    recent_data = df_features
                else:
                    logger.error("Insufficient data even for minimal training")
                    return False
            else:
                # Use only the most recent data for training
                recent_data = df_features.tail(self.lookback_days)

            # Prepare training data
            X, y = self.prepare_training_data(recent_data)

            # Remove the last row (no target available)
            X = X[:-1]
            y = y[:-1]

            if len(X) < 5:
                logger.error(f"Not enough training samples after preprocessing. Have {len(X)} samples")
                return False

            # Train model
            self.model.fit(X, y)

            logger.info(f"Model trained successfully with {len(X)} samples")
            logger.info(f"Model R² score: {self.model.score(X, y):.4f}")

            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_next_return(self, df):
        """
        Predict next day return using the trained model

        Args:
            df (pd.DataFrame): Current stock data

        Returns:
            float: Predicted return for next day
        """
        try:
            # Create features for latest data
            df_features = self.create_features(df)

            # Get the most recent data point
            latest_data = df_features.tail(1)

            feature_columns = [
                'SMA_5', 'SMA_10', 'SMA_20', 'High_Low_Ratio', 'Close_Open_Ratio',
                'Volume_Ratio', 'Price_Change', 'Price_Change_2', 'Price_Change_5', 'Volatility'
            ]

            X_latest = latest_data[feature_columns].values
            X_latest_scaled = self.scaler.transform(X_latest)

            # Make prediction
            predicted_return = self.model.predict(X_latest_scaled)[0]

            logger.info(f"Predicted next day return: {predicted_return:.4f} ({predicted_return * 100:.2f}%)")

            return predicted_return

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0

    def get_current_position(self):
        """
        Get current position for the symbol

        Returns:
            int: Current position size (positive for long, negative for short, 0 for no position)
        """
        try:
            positions = self.api.list_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return int(position.qty)
            return 0
        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return 0

    def place_order(self, side, qty):
        """
        Place a market order

        Args:
            side (str): 'buy' or 'sell'
            qty (int): Quantity to trade
        """
        try:
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f"Order placed: {side} {qty} shares of {self.symbol}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def make_trading_decision(self, predicted_return):
        """
        Make trading decision based on prediction

        Args:
            predicted_return (float): Predicted return for next day
        """
        current_position = self.get_current_position()

        logger.info(f"Current position: {current_position} shares")
        logger.info(f"Predicted return: {predicted_return * 100:.2f}%")

        if predicted_return > self.buy_threshold and current_position <= 0:
            # Buy signal
            qty_to_buy = self.position_size + abs(current_position)  # Cover short + go long
            self.place_order('buy', qty_to_buy)
            logger.info(
                f"BUY SIGNAL: Predicted return {predicted_return * 100:.2f}% > threshold {self.buy_threshold * 100:.2f}%")

        elif predicted_return < self.sell_threshold and current_position >= 0:
            # Sell signal
            qty_to_sell = self.position_size + abs(current_position)  # Close long + go short
            self.place_order('sell', qty_to_sell)
            logger.info(
                f"SELL SIGNAL: Predicted return {predicted_return * 100:.2f}% < threshold {self.sell_threshold * 100:.2f}%")

        else:
            logger.info("HOLD: No trading signal generated")

    def run_trading_cycle(self):
        """
        Run one complete trading cycle: fetch data, train model, predict, trade
        """
        logger.info(f"Starting trading cycle for {self.symbol}")

        # Fetch historical data (request more days to account for weekends/holidays)
        df = self.get_historical_data(days=90)  # Request 90 days to ensure we get enough trading days
        if df is None or len(df) < self.lookback_days:
            logger.error("Insufficient data for trading")
            return

        # Train model
        if not self.train_model(df):
            logger.error("Model training failed")
            return

        # Make prediction
        predicted_return = self.predict_next_return(df)

        # Make trading decision
        self.make_trading_decision(predicted_return)

        logger.info("Trading cycle completed")

    def run_continuous(self, interval_minutes=60):
        """
        Run the trading bot continuously

        Args:
            interval_minutes (int): Minutes between trading cycles
        """
        logger.info(f"Starting continuous trading bot for {self.symbol}")
        logger.info(f"Trading interval: {interval_minutes} minutes")

        while True:
            try:
                # Check if market is open
                clock = self.api.get_clock()
                if clock.is_open:
                    self.run_trading_cycle()
                else:
                    logger.info("Market is closed, waiting...")

                # Wait for next cycle
                time.sleep(interval_minutes * 10)

            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


# Example usage

if __name__ == "__main__":
    # Replace with your Alpaca paper trading credentials
    API_KEY = "your_alpaca_api_key"
    SECRET_KEY = "your_alpaca_secret_key"
    BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading URL

    # Initialize the trading bot
    bot = LinearRegressionTradingBot(
        api_key="PKSSRETPHT1CDMM80GCG",
        secret_key="42JYstW70pHYTJbUTwbqj5AixtXfHBqoiR937qAO",
        base_url=BASE_URL,
        symbol='AAPL'  # Change to your desired stock symbol
    )

    # Run a single trading cycle
   # bot.run_trading_cycle()

    # Or run continuously (uncomment the line below)
    bot.run_continuous(interval_minutes=60)
