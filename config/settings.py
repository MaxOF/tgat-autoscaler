CONFIG = {
    "prometheus_url": "http://localhost:9090",
    "metrics_interval": 120,      # сек (окно для /predict auto)
    "training_interval": 86400,   # сек
    "data_file": "./out/nodes.csv",
    "model_file": "./artifacts/tgat_model.pt",
    "nodes_csv_path": "./data/nodes.csv",
    "edges_csv_path": "./data/edges.csv",
    # Флаг и окно для обогащённых признаков на рёбрах
    "edge_feature_window_sec": 900, # горизонт извлечения временных рядов (напр., 15 мин)
    "edge_lag_max_minutes": 10,
    "default_edge_weight": 1,
    "edge_defaults": {
    "orders-service": { "payments-service": 0.7, "products-service": 0.6, "orders-service": 0.8 },
    "default_dst_rps": 1.0,             # дефолт для нормализации доли потока
    "default_dst_p95_ms": 250.0,        # дефолт для относительной задержки
    "edge_norm_mode": "node",           # 'node' | 'heuristic' | 'constant'
    "rps_per_core": 30.0,               # для 'heuristic': RPS на 1 vCPU (1000m)
    },
    "services": {
        "orders-service": {
            "dependencies": ["products-service", "payments-service"],
            "min_cpu": "400m",
            "max_cpu": "2000m",
            "min_memory": "512Mi",
            "max_memory": "3072Mi"
        },
        "payments-service": {
            "dependencies": [],
            "min_cpu": "400m",
            "max_cpu": "1200m",
            "min_memory": "512Mi",
            "max_memory": "2048Mi"
        },
        "products-service": {
            "dependencies": [],
            "min_cpu": "400m",
            "max_cpu": "1200m",
            "min_memory": "512Mi",
            "max_memory": "2560Mi"
        }
    }
}

PYG_AVAILABLE = True
MODEL_PATH = CONFIG.get('model_file', './artifacts/tgat_model.pt')

NODES_CSV_PATH = CONFIG.get('nodes_csv_path', 'data/nodes.csv')
EDGES_CSV_PATH = CONFIG.get('edges_csv_path', 'data/edges.csv')

# Узловые фичи — только 6 штук:
FEATURE_ORDER = [
    "cpu_mcores",   # milli-cores
    "mem_mib",      # MiB
    "rps_in",       # req/s
    "rps_out",      # req/s
    "p95_ms",       # ms
    "error_rate",   # 0..1
]

EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY= 1e-4
# (Опционально) документируем ожидаемые реберные фичи
EDGE_FEATURE_ORDER = ["edge_p95_ms", "edge_errors", "edge_rps"]


DEFAULT_MODEL_CFG = {
    'fourier_periods_min': [60, 6*60, 24*60, 7*24*60],
    'time_encoding': True,
    'dropedge_prob': 0.0,
    'd_model': 128,
    'heads': 4,
    'layers': 2,
    'dropout': 0.1,
}