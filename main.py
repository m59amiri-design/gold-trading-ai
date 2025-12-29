from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Gold Trading AI Server", version="1.0")

# ساختار داده‌های ورودی از MT5
class MarketData(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "5m"
    current_price: float
    blue_rectangle_high: Optional[float] = None
    blue_rectangle_low: Optional[float] = None
    gray_rectangle_high: Optional[float] = None
    gray_rectangle_low: Optional[float] = None

@app.get("/")
def read_root():
    return {"message": "Gold Trading AI Server is running!", "status": "active"}

@app.post("/predict")
async def predict(data: MarketData):
    """
    دریافت داده‌ها از ربات MT5 و تحلیل با هوش مصنوعی
    """
    try:
        print(f"📊 دریافت داده برای {data.symbol} - قیمت: {data.current_price}")
        
        # 1. دریافت داده‌های تاریخی طلا
        historical_data = get_gold_data()
        
        # 2. استخراج ویژگی‌های تکنیکال
        features = extract_technical_features(historical_data, data.current_price)
        
        # 3. تحلیل با الگوریتم‌های مختلف
        analysis_result = analyze_market(features, data)
        
        # 4. تولید سیگنال
        signal = generate_signal(analysis_result)
        
        return {
            "prediction": signal["direction"],
            "confidence": signal["confidence"],
            "message": signal["reason"],
            "predicted_price": signal["target_price"],
            "stop_loss": signal["stop_loss"],
            "take_profit": signal["take_profit"],
            "technical_summary": analysis_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در پردازش: {str(e)}")

def get_gold_data():
    """دریافت داده‌های تاریخی طلا از Yahoo Finance"""
    try:
        # دریافت داده‌های 60 روز گذشته با تایم‌فریم 5 دقیقه
        ticker = yf.Ticker("GC=F")  # Gold Futures
        df = ticker.history(period="60d", interval="5m")
        
        # محاسبه اندیکاتورها
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        
        return df.tail(100)  # 100 کندل آخر
    except Exception as e:
        print(f"خطا در دریافت داده: {e}")
        # در صورت خطا، داده‌های نمونه برگردان
        return create_sample_data()

def calculate_rsi(prices, period=14):
    """محاسبه RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """محاسبه MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def extract_technical_features(df, current_price):
    """استخراج ویژگی‌های تکنیکال"""
    latest = df.iloc[-1]
    
    features = {
        "price_above_sma20": current_price > latest['SMA_20'],
        "price_above_sma50": current_price > latest['SMA_50'],
        "sma20_above_sma50": latest['SMA_20'] > latest['SMA_50'],
        "rsi_value": latest['RSI'],
        "rsi_overbought": latest['RSI'] > 70,
        "rsi_oversold": latest['RSI'] < 30,
        "macd_above_signal": latest['MACD'] > latest['MACD_Signal'],
        "price_trend": "up" if current_price > df['Close'].iloc[-5] else "down",
        "volatility": df['Close'].std(),
        "volume_trend": df['Volume'].tail(5).mean() > df['Volume'].mean()
    }
    
    return features

def analyze_market(features, data):
    """تحلیل بازار با قوانین ترکیبی"""
    analysis = {
        "buy_signals": 0,
        "sell_signals": 0,
        "neutral_signals": 0,
        "reasons": []
    }
    
    # قانون 1: RSI
    if features["rsi_oversold"]:
        analysis["buy_signals"] += 1
        analysis["reasons"].append("RSI در ناحیه اشباع فروش")
    elif features["rsi_overbought"]:
        analysis["sell_signals"] += 1
        analysis["reasons"].append("RSI در ناحیه اشباع خرید")
    
    # قانون 2: میانگین‌های متحرک
    if features["price_above_sma20"] and features["sma20_above_sma50"]:
        analysis["buy_signals"] += 1
        analysis["reasons"].append("روند صعودی قوی (قیمت بالای SMA20 و SMA20 بالای SMA50)")
    
    # قانون 3: MACD
    if features["macd_above_signal"]:
        analysis["buy_signals"] += 1
        analysis["reasons"].append("MACD بالای خط سیگنال")
    
    # قانون 4: مستطیل آبی (اگر موجود باشد)
    if data.blue_rectangle_high and data.blue_rectangle_low:
        if data.current_price > data.blue_rectangle_high:
            analysis["buy_signals"] += 1
            analysis["reasons"].append("شکست مقاومت مستطیل آبی")
        elif data.current_price < data.blue_rectangle_low:
            analysis["sell_signals"] += 1
            analysis["reasons"].append("شکست حمایت مستطیل آبی")
    
    # قانون 5: روند قیمت
    if features["price_trend"] == "up":
        analysis["buy_signals"] += 0.5
    else:
        analysis["sell_signals"] += 0.5
    
    return analysis

def generate_signal(analysis):
    """تولید سیگنال نهایی"""
    total_signals = analysis["buy_signals"] + analysis["sell_signals"] + analysis["neutral_signals"]
    
    if total_signals == 0:
        return {
            "direction": "NONE",
            "confidence": 0.5,
            "reason": "عدم وجود سیگنال واضح",
            "target_price": 0,
            "stop_loss": 0,
            "take_profit": 0
        }
    
    buy_ratio = analysis["buy_signals"] / total_signals
    sell_ratio = analysis["sell_signals"] / total_signals
    
    if buy_ratio > 0.6:
        direction = "BUY"
        confidence = buy_ratio
        reason = f"سیگنال خرید قوی ({len(analysis['reasons'])} دلیل: {', '.join(analysis['reasons'])})"
    elif sell_ratio > 0.6:
        direction = "SELL"
        confidence = sell_ratio
        reason = f"سیگنال فروش قوی ({len(analysis['reasons'])} دلیل: {', '.join(analysis['reasons'])})"
    else:
        direction = "NONE"
        confidence = max(buy_ratio, sell_ratio)
        reason = "سیگنال نامشخص - منتظر تایید بیشتر"
    
    # محاسبه حد سود و ضرر
    if direction == "BUY":
        target_price = 0  # در نسخه بعدی محاسبه می‌شود
        stop_loss = 0
        take_profit = 0
    elif direction == "SELL":
        target_price = 0
        stop_loss = 0
        take_profit = 0
    else:
        target_price = 0
        stop_loss = 0
        take_profit = 0
    
    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "reason": reason,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def create_sample_data():
    """ایجاد داده‌های نمونه در صورت عدم اتصال به اینترنت"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = np.random.normal(1950, 10, 100).cumsum() + 1900
    
    df = pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    return df

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
