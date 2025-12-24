import requests
import time
import statistics

def validate_deployment():
    """Validate the deployment is working"""
    
    print("🔍 Validating NEXUS-OMEGA Deployment")
    print("=" * 40)
    
    # Test health endpoint
    try:
        health = requests.get("http://localhost:8000/api/v3/health", timeout=5)
        health.raise_for_status()
        print("✅ Health check: PASSED")
    except Exception as e:
        print(f"❌ Health check: FAILED - {e}")
        return False
    
    # Test prediction endpoint
    latencies = []
    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:8000/api/v3/predict/trading",
                headers={"Authorization": "Bearer mock_token"},
                json={"symbol": "BTC/USDT"},
                timeout=10
            )
            response.raise_for_status()
            latencies.append((time.time() - start) * 1000)
            print(f"✅ Prediction {i+1}: PASSED ({latencies[-1]:.2f}ms)")
        except Exception as e:
            print(f"❌ Prediction {i+1}: FAILED - {e}")
            return False
    
    # Performance analysis
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # Approximate p95
    
    print("\n📊 Performance Metrics:")
    print(f"P50 Latency: {p50:.2f}ms")
    print(f"P95 Latency: {p95:.2f}ms")
    print(f"Throughput: {10 / sum(latencies/1000):.2f} req/s")
    
    if p50 < 50 and p95 < 100:
        print("\n🎯 All benchmarks PASSED!")
        return True
    else:
        print("\n⚠️ Some benchmarks failed - check performance")
        return False

if __name__ == "__main__":
    validate_deployment()
