#!/bin/bash

echo "🚀 Deploying NEXUS-OMEGA Prototype"
echo "==================================="

# Build and start services
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services..."
sleep 30

# Test API
echo "Testing API..."
curl -s http://localhost:8000/api/v3/health | jq .

# Test prediction
echo "Testing prediction endpoint..."
curl -s -X POST http://localhost:8000/api/v3/predict/trading \
  -H "Authorization: Bearer mock_token" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC/USDT", "market_type": "crypto"}' | jq .

echo "✅ Setup complete!"
echo ""
echo "Access points:"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000/api/v3"
echo "Grafana: http://localhost:3001 (admin/admin123)"
echo "Prometheus: http://localhost:9090"
