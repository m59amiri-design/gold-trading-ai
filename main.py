from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Gold Trading AI", version="1.0")

class MarketData(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "5m"
    current_price: float
    blue_rectangle_high: Optional[float] = None
    blue_rectangle_low: Optional[float] = None

@app.get("/")
def home():
    return {"message": "Gold AI Server Active", "status": "online", "version": "1.0"}

@app.post("/predict")
async def predict(data: MarketData):
    try:
        # تحلیل ساده - نسخه اولیه
        signal = "NONE"
        confidence = 0.5
        
        # منطق اولیه
        if data.blue_rectangle_high and data.blue_rectangle_low:
            if data.current_price > data.blue_rectangle_high:
                signal = "BUY"
                confidence = 0.75
            elif data.current_price < data.blue_rectangle_low:
                signal = "SELL"
                confidence = 0.75
        
        return {
            "prediction": signal,
            "confidence": confidence,
            "message": f"Analyzed {data.symbol} at {data.current_price}",
            "predicted_price": data.current_price * 1.01 if signal == "BUY" else data.current_price * 0.99,
            "stop_loss": data.current_price * 0.995 if signal == "BUY" else data.current_price * 1.005,
            "take_profit": data.current_price * 1.02 if signal == "BUY" else data.current_price * 0.98
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
