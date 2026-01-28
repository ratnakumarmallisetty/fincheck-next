from fastapi import FastAPI
from api.benchmark import router as benchmark_router

app = FastAPI(
    title="Fincheck ML Benchmark API",
    description="Unified benchmarking API for MNIST and CIFAR with perturbations",
    version="1.0.0"
)

app.include_router(benchmark_router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "running"}
