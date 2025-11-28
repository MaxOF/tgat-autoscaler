from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import math
import os
import torch
import numpy as np


import pandas as pd
import torch

from app.core.prometheus import PromClient
from app.core.interfaces import GraphWindow, NodeFeatures, EdgeEvent
from config.settings import FEATURE_ORDER, CONFIG
from app.utils.parsers import now_utc_iso, parse_cpu_milli, parse_mem_mib

def build_graph_from_prometheus(config: Dict[str, Any]) -> GraphWindow:
    prom = PromClient(config['prometheus_url'])
    candidate_queries = prom.candidate_queries
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=int(config.get('metrics_interval', 120)))

    services = list(config['services'].keys())
    nodes: List[NodeFeatures] = []
  
    for svc in services:
        feats: Dict[str, float] = {k: 0.0 for k in FEATURE_ORDER}
        # Перебираем label‑ключи, пока не найдём работающий
        value_cache: Dict[str, float] = {}
        for feat_name, queries in candidate_queries.items():
     
            if feat_name not in ("cpu_mcores", "mem_mib", "rps_in", "rps_out", "p95_ms", "error_rate"):
                continue
        # cpu
        feats['cpu_mcores'] = _try_service_queries(prom, 'cpu_mcores', svc)
        # mem
        feats['mem_mib'] = _try_service_queries(prom, 'mem_mib', svc)
        # rps_in/out
        feats['rps_in'] = _try_service_queries(prom, 'rps_in', svc)
        feats['rps_out'] = _try_service_queries(prom, 'rps_out', svc)
        # p95
        feats['p95_ms'] = _try_service_queries(prom, 'p95_ms', svc)
        # errors
        val = _try_service_queries(prom, 'error_rate', svc)
        feats['error_rate'] = 0.0 if val in (None, float('inf')) else max(0.0, min(1.0, val))

        x = [float(feats[k]) for k in FEATURE_ORDER]
        nodes.append(NodeFeatures(id=svc, x=x, meta={"features": feats}))

    # Рёбра по зависимостям из CONFIG + попытка измерить трафик src→dst
    events: List[EdgeEvent] = []
    window_iso = now_utc_iso()
    tau_iso = now_utc_iso()

    node_stats: Dict[str, Dict[str, float]] = {
        n.id: (n.meta.get("features", {}) if n.meta else {}) for n in nodes
    }

    for src, scfg in config['services'].items():
        for dst in scfg.get('dependencies', []):
            edge_weight = _edge_weight_from_nodes(src, dst, node_stats, config)
            # e: кладём edge_weight в первый слот ("rps"), остальные нули как заглушки
            e = [float(edge_weight), 0.0, 0.0]
            events.append(EdgeEvent(src=src, dst=dst, tau=tau_iso, e=e, meta={'edge_weight': edge_weight}))

    gw = GraphWindow(window=window_iso, nodes=nodes, events=events, horizon=1)
    return gw


def _first_value_or_zero(result_json: Dict[str, Any]) -> float:
    try:
        data = result_json.get('data', {}).get('result', [])
        if not data:
            return 0.0
        # instant vector
        v = float(data[0]['value'][1])
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0

def _try_service_queries(prom: PromClient, key: str, svc: str) -> float:
    queries = prom.candidate_queries.get(key, [])
    for q in queries:
        for lbl in prom.serivce_label_keys:
            qq = q.replace('$SVC', svc).replace('$LBL', lbl)
            try:
                res = prom.query(qq)
                v = _first_value_or_zero(res)
                if v != 0.0:
                    return v
            except Exception:
                continue
    # если ничего не нашли — 0
    return 0.0

def _edge_weight_from_nodes(src: str, dst: str, node_stats: Dict[str, Dict[str, float]], config: Dict[str, Any]) -> float:
    # Базовый вес: явный из edge_defaults, иначе равномерно по исходящим рёбрам, иначе default_edge_weight/0.5
    out_deg = len(config['services'].get(src, {}).get('dependencies', [])) or 0
    base_w = config.get('edge_defaults', {}).get(src, {}).get(dst)
    if base_w is None:
        base_w = (1.0 / out_deg) if out_deg > 0 else config.get('default_edge_weight', 1)


    # Источники признаков
    sf = node_stats.get(src, {})
    df = node_stats.get(dst, {})


    # Нормировки/рефы
    rps_norm_src = float(config.get('rps_norm_src', 50.0))
    rps_norm_dst = float(config.get('rps_norm_dst', 50.0))
    p95_ref_ms = float(config.get('p95_ref_ms', config.get('default_dst_p95_ms', 250.0)))


    # Нагрузочные факторы
    src_pressure = 0.0
    if rps_norm_src > 0:
        src_pressure = min(1.0, float(sf.get('rps_out', 0.0)) / rps_norm_src)


    dst_flow = 0.0
    if rps_norm_dst > 0:
        dst_flow = min(1.0, float(df.get('rps_in', 0.0)) / rps_norm_dst)


    # Задержка и надёжность назначения
    p95 = float(df.get('p95_ms', 0.0))
    lat_factor = (p95 / (p95 + p95_ref_ms)) if p95 > 0 else 0.0 # 0..1, растёт с ростом p95
    reliability = 1.0 - max(0.0, min(1.0, float(df.get('error_rate', 0.0))))


    # Загруженность CPU/MEM назначения относительно max из CONFIG
    dst_cfg = CONFIG['services'].get(dst, {})
    max_cpu_m = parse_cpu_milli(dst_cfg.get('max_cpu', '2000m')) or 1.0
    max_mem_m = parse_mem_mib(dst_cfg.get('max_memory', '2048Mi')) or 1.0
    u_cpu = min(1.0, float(df.get('cpu_mcores', 0.0)) / max_cpu_m)
    u_mem = min(1.0, float(df.get('mem_mib', 0.0)) / max_mem_m)
    dst_load = max(u_cpu, u_mem)


    # Веса компонентов (можно вынести в CONFIG)
    alpha = float(config.get('ew_alpha', 0.4)) # базовый
    beta = float(config.get('ew_beta', 0.2)) # давление источника
    gamma = float(config.get('ew_gamma', 0.2)) # входной поток/нагрузка назначения
    delta = float(config.get('ew_delta', 0.2)) # задержка назначения


    # Итоговый вес 0..1
    w = alpha*base_w + beta*src_pressure + gamma*max(dst_flow, dst_load) + delta*lat_factor
    w = max(0.0, min(1.0, w))
    # штраф по надёжности
    w *= reliability
    return float(max(0.0, min(1.0, w)))

def edge_weight_from_nodes(src: str, dst: str, node_stats: Dict[str, Dict[str, float]], cfg: Dict[str, Any]) -> float:
    out_deg = len(CONFIG['services'].get(src, {}).get('dependencies', []))
    base_w = cfg.get('edge_defaults', {}).get(src, {}).get(dst)
    if base_w is None:
        base_w = (1.0/out_deg) if out_deg > 0 else cfg.get('default_edge_weight', 0.5)

    sf = node_stats.get(src, {}); df = node_stats.get(dst, {})
    rps_norm_src = float(cfg.get('rps_norm_src', 120.0))
    rps_norm_dst = float(cfg.get('rps_norm_dst', 120.0))
    p95_ref_ms   = float(cfg.get('p95_ref_ms', 250.0))

    src_pressure = min(1.0, (sf.get('rps_out', 0.0)/rps_norm_src) if rps_norm_src > 0 else 0.0)
    dst_flow     = min(1.0, (df.get('rps_in', 0.0)/rps_norm_dst) if rps_norm_dst > 0 else 0.0)

    p95 = df.get('p95_ms', 0.0)
    lat_factor = (p95/(p95 + p95_ref_ms)) if p95 > 0 else 0.0
    reliability = 1.0 - max(0.0, min(1.0, df.get('error_rate', 0.0)))

    dst_cfg = CONFIG['services'].get(dst, {})
    max_cpu_m = parse_cpu_milli(dst_cfg.get('max_cpu', '2000m')) or 1.0
    max_mem_m = parse_mem_mib(dst_cfg.get('max_memory', '2048Mi')) or 1.0
    u_cpu = min(1.0, (df.get('cpu_mcores', 0.0)/max_cpu_m))
    u_mem = min(1.0, (df.get('mem_mib', 0.0)/max_mem_m))
    dst_load = max(u_cpu, u_mem)

    a = float(cfg.get('ew_alpha', 0.4))
    b = float(cfg.get('ew_beta', 0.2))
    g = float(cfg.get('ew_gamma', 0.2))
    d = float(cfg.get('ew_delta', 0.2))

    w = a*base_w + b*src_pressure + g*max(dst_flow, dst_load) + d*lat_factor
    w = max(0.0, min(1.0, w)) * reliability
    return float(max(0.0, min(1.0, w)))


# def _graphs_from_csv(model: TGATAutoscalerModel, nodes_csv_path: str, edges_csv_path: Optional[str], time_col: str = "window_utc"):
#     by_t = defaultdict(lambda: {'nodes': {}, 'targets': {}})
#     with open(nodes_csv_path, newline='', encoding='utf-8') as f:
#         rdr = csv.DictReader(f)
#         for row in rdr:
#             t = row[time_col]
#             sid = row['service']
#             feats = {
#                 'cpu_mcores': float(row['cpu_mcores']),
#                 'mem_mib': float(row['mem_mib']),
#                 'rps_in': float(row['rps_in']),
#                 'rps_out': float(row['rps_out']),
#                 'p95_ms': float(row['p95_ms']),
#                 'error_rate': float(row['error_rate']),
#             }
#             by_t[t]['nodes'][sid] = feats
#             by_t[t]['targets'][sid] = {
#                 'replicas': int(row.get('target_replicas', 1)),
#                 'cpu_m': float(row.get('target_cpu_m', feats['cpu_mcores'])),
#                 'mem_mib': float(row.get('target_mem_mib', feats['mem_mib'])),
#             }

#     # читаем edges.csv (если есть)
#     edges_by_t = defaultdict(list)
#     if edges_csv_path and os.path.exists(edges_csv_path):
#         with open(edges_csv_path, newline='', encoding='utf-8') as f:
#             rdr = csv.DictReader(f)
#             for row in rdr:
#                 t = row[time_col]
#                 edges_by_t[t].append((row['src'], row['dst'], float(row['edge_weight'])))

#     # соберём графы
#     data = []
#     for win_iso, bundle in by_t.items():
#         nodes = []
#         # фиксируем порядок по CONFIG.services
#         for sid in CONFIG['services'].keys():
#             feats = bundle['nodes'].get(sid, {k:0.0 for k in FEATURE_ORDER})
#             x = [float(feats[k]) for k in FEATURE_ORDER]
#             nodes.append(NodeFeatures(id=sid, x=x, meta={'features': feats}))

#         # события (рёбра)
#         events = []
#         # если нет csv с рёбрами — считаем edge_weight по узлам и зависимостям
#         if edges_by_t.get(win_iso):
#             for src, dst, w in edges_by_t[win_iso]:
#                 tau = (parse_iso8601(win_iso) - timedelta(seconds=60)).replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')
#                 events.append(EdgeEvent(src=src, dst=dst, tau=tau, e=[w, 0.0, 0.0]))
#         else:
#             node_stats = bundle['nodes']
#             ew_cfg = {
#                 'default_edge_weight': CONFIG.get('default_edge_weight', 0.5),
#                 'edge_defaults': CONFIG.get('edge_defaults', {}),
#                 'rps_norm_src': CONFIG.get('rps_norm_src', 120.0),
#                 'rps_norm_dst': CONFIG.get('rps_norm_dst', 120.0),
#                 'p95_ref_ms': CONFIG.get('p95_ref_ms', 250.0),
#                 'ew_alpha': CONFIG.get('ew_alpha', 0.4),
#                 'ew_beta': CONFIG.get('ew_beta', 0.2),
#                 'ew_gamma': CONFIG.get('ew_gamma', 0.2),
#                 'ew_delta': CONFIG.get('ew_delta', 0.2),
#             }
#             for src, scfg in CONFIG['services'].items():
#                 for dst in scfg.get('dependencies', []):
#                     w = edge_weight_from_nodes(src, dst, node_stats, ew_cfg)
#                     tau = (parse_iso8601(win_iso) - timedelta(seconds=60)).replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')
#                     events.append(EdgeEvent(src=src, dst=dst, tau=tau, e=[w, 0.0, 0.0]))

#         gw = GraphWindow(window=win_iso, nodes=nodes, events=events, horizon=1)

#         # цели
#         target_list = []
#         for sid in CONFIG['services'].keys():
#             tgt = bundle['targets'].get(sid, {'replicas':1,'cpu_m':500.0,'mem_mib':512.0})
#             target_list.append({'id': sid, **tgt})

#         # превратим в (x, edge_index, edge_attr, y) как раньше
#         x, edge_index, edge_attr = model.build_graph_from_payload(gw)
#         Y = []
#         for sid in model.idx2id:
#             t = next((t for t in target_list if t['id']==sid), None)
#             if t is None:
#                 Y.append((1.0, 500.0, 512.0))
#             else:
#                 Y.append((float(t['replicas']), float(t['cpu_m']), float(t['mem_mib'])))
#         y = torch.tensor(np.asarray(Y, dtype=np.float32))
#         data.append((x, edge_index, edge_attr, y))
#     return data






# Ожидается, что следующие объекты уже есть в вашем проекте:
# - CONFIG (из вашего сообщения)
# - FEATURE_ORDER (порядок признаков узла, может содержать и edge_*; см. фильтрацию ниже)
# - TGATAutoscalerModel, NodeFeatures, EdgeEvent, GraphWindow
# - edge_weight_from_nodes(src, dst, node_stats: Dict[str, Dict[str, float]], ew_cfg: Dict) -> float

def _graphs_from_csv(
    model,
    nodes_csv_path: str,
    edges_csv_path: Optional[str],
    time_col: str = "window_utc"
):
    """
    Сборка датасета графов по окнам времени из CSV-файлов с помощью pandas.

    nodes.csv — ожидаемые колонки:
      - time_col (по умолчанию 'window_utc'): ISO-8601 или парсибельный timestamp
      - service: идентификатор сервиса (должен совпадать с ключами в CONFIG["services"])
      - cpu_mcores, mem_mib, rps_in, rps_out, p95_ms, error_rate
      - опционально: target_replicas, target_cpu_m, target_mem_mib
      - допускаются дополнительные фичи; будут взяты только пересечения с FEATURE_ORDER

    edges.csv — опционально, ожидаемые колонки:
      - time_col (тот же, что и в nodes.csv)
      - src, dst
      - опционально: edge_weight, edge_p95_ms, edge_errors, edge_rps
        (если нет — веса/атрибуты будут синтезированы по узловым метрикам и CONFIG)
    """
    # ---------- 1) Чтение и предобработка узлов ----------
    if not os.path.exists(nodes_csv_path):
        raise FileNotFoundError(f"nodes_csv_path not found: {nodes_csv_path}")

    nodes_df = pd.read_csv(nodes_csv_path)
    if time_col not in nodes_df.columns:
        raise ValueError(f"Column '{time_col}' is missing in nodes CSV")

    # Приводим время к UTC и сортируем
    nodes_df[time_col] = pd.to_datetime(nodes_df[time_col], utc=True, errors="coerce")
    if nodes_df[time_col].isna().any():
        bad = nodes_df[nodes_df[time_col].isna()]
        raise ValueError(f"Unparseable timestamps in '{time_col}':\n{bad[[time_col, 'service']].head()}")

    nodes_df = nodes_df.sort_values(time_col)

    # Нормализуем названия узловых колонок и приводим к float
    # Возьмём только те фичи, которые реально есть в файле
    # и пересекаются с FEATURE_ORDER.
    available_cols = set(nodes_df.columns)
    node_feature_cols = [c for c in FEATURE_ORDER if c in available_cols]
    # Если FEATURE_ORDER содержит edge_* (ваш список), не включаем их в узловой вектор:
    node_feature_cols = [c for c in node_feature_cols if not c.startswith("edge_")]

    # Приведение типов для узловых метрик (где они есть)
    for col in ["cpu_mcores", "mem_mib", "rps_in", "rps_out", "p95_ms", "error_rate"]:
        if col in nodes_df.columns:
            nodes_df[col] = pd.to_numeric(nodes_df[col], errors="coerce").fillna(0.0)

    # Цели (если есть), иначе дефолты
    if "target_replicas" in nodes_df.columns:
        nodes_df["target_replicas"] = pd.to_numeric(nodes_df["target_replicas"], errors="coerce").fillna(1).astype(int)
    else:
        nodes_df["target_replicas"] = 1

    if "target_cpu_m" in nodes_df.columns:
        nodes_df["target_cpu_m"] = pd.to_numeric(nodes_df["target_cpu_m"], errors="coerce").fillna(nodes_df["cpu_mcores"].fillna(500.0))
    else:
        nodes_df["target_cpu_m"] = nodes_df.get("cpu_mcores", pd.Series([], dtype=float)).fillna(500.0)

    if "target_mem_mib" in nodes_df.columns:
        nodes_df["target_mem_mib"] = pd.to_numeric(nodes_df["target_mem_mib"], errors="coerce").fillna(nodes_df["mem_mib"].fillna(512.0))
    else:
        nodes_df["target_mem_mib"] = nodes_df.get("mem_mib", pd.Series([], dtype=float)).fillna(512.0)

    # Делаем pivot по сервисам внутри окна, чтобы удобнее было смотреть метрики по сервису
    # но для совместимости ниже будем собирать словари.
    # Гарантируем наличие столбца 'service'
    if "service" not in nodes_df.columns:
        raise ValueError("Column 'service' is missing in nodes CSV")

    # ---------- 2) Чтение и предобработка рёбер (опционально) ----------
    edges_df = None
    has_edges_csv = bool(edges_csv_path and os.path.exists(edges_csv_path))
    if has_edges_csv:
        edges_df = pd.read_csv(edges_csv_path)
        if time_col not in edges_df.columns:
            raise ValueError(f"Column '{time_col}' is missing in edges CSV")
        # Привести к UTC
        edges_df[time_col] = pd.to_datetime(edges_df[time_col], utc=True, errors="coerce")
        if edges_df[time_col].isna().any():
            bad_e = edges_df[edges_df[time_col].isna()]
            raise ValueError(f"Unparseable timestamps in edges '{time_col}':\n{bad_e[[time_col, 'src', 'dst']].head()}")
        # Типы
        for ecol in ["edge_weight", "edge_rps", "edge_p95_ms", "edge_errors"]:
            if ecol in edges_df.columns:
                edges_df[ecol] = pd.to_numeric(edges_df[ecol], errors="coerce").fillna(0.0)
        # Фильтруем на разумные src/dst
        for col in ["src", "dst"]:
            if col not in edges_df.columns:
                raise ValueError(f"Column '{col}' is missing in edges CSV")

    # Удобные срезы
    service_order: List[str] = list(CONFIG["services"].keys())

    # Настройки для синтеза рёбер
    ew_cfg = {
        "default_edge_weight": CONFIG.get("default_edge_weight", 0.5),
        "edge_defaults": CONFIG.get("edge_defaults", {}),
        "rps_norm_src": CONFIG.get("rps_norm_src", 120.0),
        "rps_norm_dst": CONFIG.get("rps_norm_dst", 120.0),
        "p95_ref_ms": CONFIG.get("p95_ref_ms", 250.0),
        "ew_alpha": CONFIG.get("ew_alpha", 0.4),
        "ew_beta": CONFIG.get("ew_beta", 0.2),
        "ew_gamma": CONFIG.get("ew_gamma", 0.2),
        "ew_delta": CONFIG.get("ew_delta", 0.2),
    }

    # ---------- 3) Группируем по окнам и собираем графы ----------
    data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    # Группировка узлов по окну
    for win_ts, win_df in nodes_df.groupby(time_col, sort=True):
        # Словарь узловых метрик по сервису для синтеза рёбер
        node_stats: Dict[str, Dict[str, float]] = {}
        targets_stats: Dict[str, Dict[str, float]] = {}

        # Собираем метрики по сервисам в текущем окне
        for _, row in win_df.iterrows():
            sid = str(row["service"])
            feats_row = {
                "cpu_mcores": float(row.get("cpu_mcores", 0.0)),
                "mem_mib": float(row.get("mem_mib", 0.0)),
                "rps_in": float(row.get("rps_in", 0.0)),
                "rps_out": float(row.get("rps_out", 0.0)),
                "p95_ms": float(row.get("p95_ms", 0.0)),
                "error_rate": float(row.get("error_rate", 0.0)),
            }
            node_stats[sid] = feats_row
            targets_stats[sid] = {
                "replicas": int(row.get("target_replicas", 1)),
                "cpu_m": float(row.get("target_cpu_m", feats_row["cpu_mcores"] or 500.0)),
                "mem_mib": float(row.get("target_mem_mib", feats_row["mem_mib"] or 512.0)),
            }

        # Собираем NodeFeatures в порядке CONFIG["services"]
        nodes: List["NodeFeatures"] = []
        for sid in service_order:
            feats = node_stats.get(sid, None)
            # x-ветор формируем только из доступных узловых колонок (без edge_*)
            if feats is None:
                # сервис отсутствует в окне — заполняем нулями
                x_vals = [0.0 for _ in node_feature_cols] if node_feature_cols else [
                    # бэкап: классический порядок
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
                meta_feats = {c: 0.0 for c in node_feature_cols} if node_feature_cols else {}
            else:
                meta_feats = feats.copy()
                # Пересечение колонок + порядок
                if node_feature_cols:
                    x_vals = [float(meta_feats.get(c, 0.0)) for c in node_feature_cols]
                else:
                    # Бэкап порядок
                    x_vals = [
                        float(meta_feats.get("cpu_mcores", 0.0)),
                        float(meta_feats.get("mem_mib", 0.0)),
                        float(meta_feats.get("rps_in", 0.0)),
                        float(meta_feats.get("rps_out", 0.0)),
                        float(meta_feats.get("p95_ms", 0.0)),
                        float(meta_feats.get("error_rate", 0.0)),
                    ]

            nodes.append(NodeFeatures(id=sid, x=x_vals, meta={"features": meta_feats}))

        # ---------- Рёбра (events) ----------
        events: List["EdgeEvent"] = []
        # tau = время события (на минуту раньше, как в вашем коде)
        tau_ts = (pd.Timestamp(win_ts).to_pydatetime() - timedelta(seconds=60)).replace(tzinfo=timezone.utc)
        tau_iso = tau_ts.isoformat().replace("+00:00", "Z")

        def _edge_attr_from_sources(src: str, dst: str, w: float) -> List[float]:
            """Собираем e=[w, l_tilde, eps_tilde] как в System Model."""
            dst_p95 = float(node_stats.get(dst, {}).get("p95_ms", 0.0))
            dst_err = float(node_stats.get(dst, {}).get("error_rate", 0.0))
            l_tilde = w * dst_p95
            eps_tilde = w * dst_err
            return [float(w), float(l_tilde), float(eps_tilde)]

        if has_edges_csv:
            # Рёбра доступны из файла → берём всё, что есть, иначе синтезируем w
            cur_edges = edges_df[edges_df[time_col] == win_ts]
            # Бывает, что в этом окне нет ни одной строки — fallback на синтез
            if len(cur_edges) > 0:
                for _, er in cur_edges.iterrows():
                    src = str(er["src"])
                    dst = str(er["dst"])
                    # Если в CSV нет веса — синтезируем
                    if "edge_weight" in er and pd.notna(er["edge_weight"]) and float(er["edge_weight"]) > 0.0:
                        w = float(er["edge_weight"])
                    else:
                        w = edge_weight_from_nodes(src, dst, node_stats, ew_cfg)

                    # Если даны edge_p95_ms / edge_errors — используем их, иначе пропагируем от dst
                    if "edge_p95_ms" in er and pd.notna(er["edge_p95_ms"]):
                        l_tilde = float(er["edge_p95_ms"])
                    else:
                        l_tilde = w * float(node_stats.get(dst, {}).get("p95_ms", 0.0))

                    if "edge_errors" in er and pd.notna(er["edge_errors"]):
                        eps_tilde = float(er["edge_errors"])
                    else:
                        eps_tilde = w * float(node_stats.get(dst, {}).get("error_rate", 0.0))

                    e_vec = [w, l_tilde, eps_tilde]
                    events.append(EdgeEvent(src=src, dst=dst, tau=tau_iso, e=e_vec))
            else:
                # Фаллбэк: синтез по конфигу и узловым метрикам
                for src, scfg in CONFIG["services"].items():
                    for dst in scfg.get("dependencies", []):
                        w = edge_weight_from_nodes(src, dst, node_stats, ew_cfg)
                        e_vec = _edge_attr_from_sources(src, dst, w)
                        events.append(EdgeEvent(src=src, dst=dst, tau=tau_iso, e=e_vec))
        else:
            # Рёберный CSV отсутствует → синтезируем из узловых метрик
            for src, scfg in CONFIG["services"].items():
                for dst in scfg.get("dependencies", []):
                    w = edge_weight_from_nodes(src, dst, node_stats, ew_cfg)
                    e_vec = _edge_attr_from_sources(src, dst, w)
                    events.append(EdgeEvent(src=src, dst=dst, tau=tau_iso, e=e_vec))

        # ---------- GraphWindow и таргеты ----------
        gw = GraphWindow(window=pd.Timestamp(win_ts).isoformat().replace("+00:00", "Z"),
                         nodes=nodes, events=events, horizon=1)

        # Список целей в порядке service_order
        target_list = []
        for sid in service_order:
            trow = targets_stats.get(sid, None)
            if trow is None:
                target_list.append({"id": sid, "replicas": 1, "cpu_m": 500.0, "mem_mib": 512.0})
            else:
                target_list.append({"id": sid,
                                    "replicas": float(trow["replicas"]),
                                    "cpu_m": float(trow["cpu_m"]),
                                    "mem_mib": float(trow["mem_mib"])})

        # ---------- В тензоры через модельный конструктор ----------
        x, edge_index, edge_attr = model.build_graph_from_payload(gw)

        # y — в порядке model.idx2id
        Y = []
        for sid in model.idx2id:
            t = next((t for t in target_list if t["id"] == sid), None)
            if t is None:
                Y.append((1.0, 500.0, 512.0))
            else:
                Y.append((float(t["replicas"]), float(t["cpu_m"]), float(t["mem_mib"])))
        y = torch.tensor(np.asarray(Y, dtype=np.float32))

        data.append((x, edge_index, edge_attr, y))

    return data
