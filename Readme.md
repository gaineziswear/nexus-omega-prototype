# 1. Clone the structure
git clone https://github.com/your-org/nexus-omega-prototype.git
cd nexus-omega-prototype

# 2. Make scripts executable
chmod +x scripts/deploy.sh
chmod +x scripts/validation.py

# 3. Deploy everything
./scripts/deploy.sh

# 4. Validate
python scripts/validation.py

# 5. Open browser
# Frontend: http://localhost:3000
# Grafana: http://localhost:3001 (login: admin/admin123)

# 6. View logs
docker-compose logs -f api-gateway
docker-compose logs -f frontend
