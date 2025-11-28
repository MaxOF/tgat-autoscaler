# TGAT-Autoscaler

> Темпоральный графовый автоскейлер для Kubernetes (TGAT + GATv2): объединяет реактивные метрики и проактивное прогнозирование для горизонтального **и** вертикального масштабирования микросервисов.


## TL;DR

- **Граф зависимостей** сервисов + **временные кодировки** → модель TGAT предсказывает `[реплики, CPU (m), память (MiB)]` для каждого сервиса.
- **Нет метрик на рёбрах?** Строим **суррогатные** веса по узловым метрикам (RPS, p95, ошибки) и топологии.
- **Безопасность**: гистерезис, лимиты скорости, проекция в допустимые диапазоны, корректное применение к K8s (масштабирование и по репликам, и по ресурсам).
- **Интерфейс**: FastAPI — `/train` (CSV), `/predict` (Prometheus/JSON), `/apply` (масштабирование в кластер).

---

## Архитектура

```mermaid
flowchart LR
  subgraph Cluster[Kubernetes Cluster]
    subgraph App[Microservices]
      A[orders-service]
      B[products-service]
      C[payments-service]
    end
    P[(Prometheus)]
    G[(kube-state-metrics / cAdvisor)]
  end

  subgraph Service[TGAT-Autoscaler Service (FastAPI)]
    D[Build Graph\n(node features + edge synth)]
    E[TGAT Predictor\n(GATv2 + Time Encoding)]
    F[Safety Policy\n(hysteresis, rate caps, bounds)]
    H[/train from CSV]
    I[/predict from Prom/JSON]
    J[/apply to K8s]
  end

  A -- HTTP/gRPC --> B
  A -- HTTP/gRPC --> C
  G --> P
  App --> G
  P --> D
  I --> D --> E --> F --> J
  H --> E
