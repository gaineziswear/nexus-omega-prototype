import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar
} from 'recharts';
import './App.css';

const API_URL = 'http://localhost:8000/api/v3';

function Dashboard() {
  const [signals, setSignals] = useState([]);
  const [marketData, setMarketData] = useState([]);
  const [metrics, setMetrics] = useState({
    accuracy: 87.2,
    latency: 23,
    activeSignals: 156
  });

  useEffect(() => {
    // Fetch initial data
    fetchPrediction();
    
    // Set up WebSocket simulation
    const interval = setInterval(() => {
      generateMockSignal();
    }, 5000); // Every 5 seconds
    
    return () => clearInterval(interval);
  }, []);

  const fetchPrediction = async () => {
    try {
      const response = await fetch(`${API_URL}/predict/trading`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer mock_token'
        },
        body: JSON.stringify({
          symbol: 'BTC/USDT',
          timeframe: '1m',
          market_type: 'crypto'
        })
      });
      
      const data = await response.json();
      setSignals(prev => [data, ...prev.slice(0, 9)]);
      
      // Add to chart
      setMarketData(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        price: data.target_price,
        confidence: data.confidence * 100
      }].slice(-20));
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  const generateMockSignal = () => {
    const mockSignal = {
      prediction_id: `mock_${Date.now()}`,
      symbol: ['BTC/USDT', 'ETH/USDT', 'AAPL'][Math.floor(Math.random() * 3)],
      action: ['BUY', 'SELL'][Math.floor(Math.random() * 2)],
      confidence: Math.random() * 0.3 + 0.7,
      target_price: 40000 + Math.random() * 5000,
      stop_loss: 38000 + Math.random() * 4000,
      expected_return: (Math.random() - 0.5) * 0.1,
      timestamp: new Date().toISOString()
    };
    
    setSignals(prev => [mockSignal, ...prev.slice(0, 9)]);
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>NEXUS-OMEGA v3.0</h1>
        <div className="metrics">
          <div className="metric">
            <span className="value">{metrics.accuracy}%</span>
            <span className="label">Accuracy</span>
          </div>
          <div className="metric">
            <span className="value">{metrics.latency}ms</span>
            <span className="label">Latency</span>
          </div>
          <div className="metric">
            <span className="value">{metrics.activeSignals}</span>
            <span className="label">Active Signals</span>
          </div>
        </div>
      </header>

      <div className="main-content">
        <section className="signals-section">
          <h2>Live AI Signals</h2>
          {signals.map(signal => (
            <div key={signal.prediction_id} className={`signal-card ${signal.action.toLowerCase()}`}>
              <div className="signal-header">
                <span className="symbol">{signal.symbol}</span>
                <span className="confidence">
                  {(signal.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="signal-body">
                <span className="action">{signal.action}</span>
                <span className="price">Target: ${signal.target_price?.toFixed(2)}</span>
                <span className="stop">Stop: ${signal.stop_loss?.toFixed(2)}</span>
              </div>
              <div className="signal-footer">
                {new Date(signal.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </section>

        <section className="charts-section">
          <h2>Market Overview</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={marketData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>

          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={marketData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="confidence" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <button className="refresh-btn" onClick={fetchPrediction}>
        Generate New Signal
      </button>
    </div>
  );
}

export default Dashboard;
