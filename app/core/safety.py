from typing import Dict, Any, List
import os
import json
from fastapi import HTTPException
from kubernetes import client, config
import numpy as np

from app.core.interfaces import Action, PolicyConfig
from app.utils.parsers import parse_cpu_milli, parse_mem_mib, format_cpu_milli, format_mem_gi_from_mib
from config.settings import CONFIG


class SafetyPolicy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self.memory_path = os.environ.get('TGAT_STATE_PATH', './tgat_state.json')
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        try:
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def filter(self, actions: List[Action], tstamp: str) -> List[Action]:
        hist_w = self.cfg.hysteresis_windows
        rate_lim = self.cfg.rate_limit_replicas
        rmin, rmax = self.cfg.r_min, self.cfg.r_max

        prev_actions: Dict[str, Dict[str, Any]] = self.state.get('last_actions', {})
        counters: Dict[str, int] = self.state.get('hysteresis', {})
        new_actions: List[Action] = []

        for a in actions:
            prev = prev_actions.get(a.id)
            if prev is None or (prev['replicas'] != a.replicas or prev.get('cpu') != a.cpu or prev.get('mem') != a.mem):
                cnt = counters.get(a.id, 0) + 1
                counters[a.id] = cnt
                if cnt < hist_w:
                    if prev is not None:
                        a = Action(id=a.id, replicas=prev['replicas'], cpu=prev.get('cpu'), mem=prev.get('mem'))
                else:
                    counters[a.id] = 0
            else:
                counters[a.id] = 0

            if prev is not None:
                dr = a.replicas - int(prev['replicas'])
                if abs(dr) > rate_lim:
                    a = Action(id=a.id, replicas=int(prev['replicas']) + (rate_lim if dr > 0 else -rate_lim), cpu=a.cpu, mem=a.mem)

            a = Action(id=a.id, replicas=int(max(rmin, min(rmax, a.replicas))), cpu=a.cpu, mem=a.mem)

            # Доп. клиппинг ресурсов по CONFIG
            svc_cfg = CONFIG['services'].get(a.id, {})
            min_cpu = parse_cpu_milli(svc_cfg.get('min_cpu', '100m'))
            max_cpu = parse_cpu_milli(svc_cfg.get('max_cpu', '4000m'))
            min_mem = parse_mem_mib(svc_cfg.get('min_memory', '256Mi'))
            max_mem = parse_mem_mib(svc_cfg.get('max_memory', '8192Mi'))
            if a.cpu:
                cpu_v = parse_cpu_milli(a.cpu)
                a.cpu = format_cpu_milli(float(np.clip(cpu_v, min_cpu, max_cpu)))
            if a.mem:
                mem_v = parse_mem_mib(a.mem)
                a.mem = format_mem_gi_from_mib(float(np.clip(mem_v, min_mem, max_mem)))

            new_actions.append(a)

        self.state['last_window'] = tstamp
        self.state['last_actions'] = {a.id: {'replicas': a.replicas, 'cpu': a.cpu, 'mem': a.mem} for a in new_actions}
        self.state['hysteresis'] = counters
        self._save_state()
        return new_actions


    def apply_to_k8s(self, actions: List[Action]) -> Dict[str, Any]:
        """
        Применяет решения автоскейлера в Kubernetes:
        1) ВЕРТИКАЛЬ: обновляет requests/limits CPU/Memory через Server-Side Apply (SSA).
        - По умолчанию патчит все контейнеры в поде.
        - Если задан TGAT_CONTAINER, патчит только его.
        - Использует field_manager=tgat-autoscaler (переопределяемый переменной).
        - Если SSA недоступен, делает fallback на strategic-merge patch.
        2) ГОРИЗОНТАЛЬ: патчит scale.subresource для Deployment (реплики).
        """
        # dry-run режим
        if getattr(self.cfg, "dry_run", False) or bool(self.cfg.get("dry_run")):
            return {"dry_run": True, "applied": [a.dict() for a in actions]}

        # kubeconfig
        try:
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                config.load_incluster_config()
            else:
                config.load_kube_config()
            apps = client.AppsV1Api()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"K8s init failed: {e}")

        namespace = os.getenv("TGAT_NAMESPACE", "default")
        field_manager = os.getenv("TGAT_FIELD_MANAGER", "tgat-autoscaler")
        target_container = os.getenv("TGAT_CONTAINER")  # если хотим патчить конкретный контейнер

        report = {"dry_run": False, "patched": [], "details": []}

        for a in actions:
            dep_name = a.id
            desired_cpu = str(a.cpu)   # например "750m"
            desired_mem = str(a.mem)   # например "1Gi"

            item_result = {"deployment": dep_name, "resources": None, "replicas": None, "errors": []}

            # ---------- 1) ВЕРТИКАЛЬ: ресурсы (SSA с fallback) ----------
            try:
                # Узнаем список контейнеров (если нужно патчить все)
                containers = []
                if target_container:
                    containers = [target_container]
                else:
                    try:
                        dep = apps.read_namespaced_deployment(dep_name, namespace)
                        containers = [c.name for c in dep.spec.template.spec.containers]
                    except Exception:
                        # если не смогли прочитать deployment — патчим контейнер с тем же именем, что и deployment
                        containers = [dep_name]

                containers_patch = [{
                    "name": c_name,
                    "resources": {
                        "requests": {"cpu": desired_cpu, "memory": desired_mem},
                        "limits":   {"cpu": desired_cpu, "memory": desired_mem},
                    }
                } for c_name in containers]

                ssa_patch = {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {"name": dep_name, "namespace": namespace},
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": containers_patch
                            }
                        }
                    }
                }

                # Пытаемся через Server-Side Apply (JSON)
                apps.patch_namespaced_deployment(
                    name=dep_name,
                    namespace=namespace,
                    body=ssa_patch,
                    field_manager=field_manager,
                    force=True,
                    _content_type="application/apply-patch+json",
                )
                item_result["resources"] = {"mode": "ssa", "containers": containers}
            except HTTPException as e:
                # fallback: strategic-merge patch
                try:
                    apps.patch_namespaced_deployment(
                        name=dep_name,
                        namespace=namespace,
                        body=ssa_patch,  # тот же dict годится
                        _content_type="application/strategic-merge-patch+json",
                    )
                    item_result["resources"] = {"mode": "strategic-merge", "containers": containers, "note": "SSA fallback"}
                except Exception as e2:
                    item_result["errors"].append(f"resources_patch_failed: {e2}")

            # ---------- 2) ГОРИЗОНТАЛЬ: scale.subresource ----------
            try:
                scale_body = {"spec": {"replicas": int(a.replicas)}}
                apps.patch_namespaced_deployment_scale(
                    name=dep_name,
                    namespace=namespace,
                    body=scale_body
                )
                item_result["replicas"] = int(a.replicas)
            except Exception as e:
                item_result["errors"].append(f"scale_patch_failed: {e}")

            # ----------
            report["patched"].append(dep_name)
            report["details"].append(item_result)

        return report

