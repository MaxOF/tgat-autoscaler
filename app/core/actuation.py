"""Kubernetes adapter; no prediction or policy selection."""

from __future__ import annotations
import os
from typing import Any, Dict, List
from fastapi import HTTPException
from kubernetes import client, config
from app.core.interfaces import Action, PolicyConfig


class KubernetesActuator:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

    def apply(self, actions: List[Action]) -> Dict[str, Any]:
        """
        Apply horizontal and vertical actions.

        Vertical:
            Patch CPU/memory requests and limits.

        Horizontal:
            Patch Deployment scale subresource.
        """

        if self.cfg.dry_run:
            return {
                "dry_run": True,
                "applied": [action.model_dump() for action in actions],
            }

        try:
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                config.load_incluster_config()

            else:
                config.load_kube_config()

            apps = client.AppsV1Api()

        except Exception as exc:
            raise HTTPException(status_code=500, detail=("K8s init failed: " f"{exc}"))

        namespace = os.getenv("TGAT_NAMESPACE", "default")

        field_manager = os.getenv("TGAT_FIELD_MANAGER", "tgat-autoscaler")

        target_container = os.getenv("TGAT_CONTAINER")

        report: Dict[str, Any] = {"dry_run": False, "patched": [], "details": []}

        for action in actions:
            dep_name = action.id

            item_result = {
                "deployment": dep_name,
                "resources": None,
                "replicas": None,
                "errors": [],
            }

            desired_cpu = action.cpu
            desired_mem = action.mem

            # -----------------------------------------------------------------
            # Vertical scaling
            # -----------------------------------------------------------------
            if desired_cpu or desired_mem:
                try:
                    if target_container:
                        containers = [target_container]

                    else:
                        try:
                            deployment = apps.read_namespaced_deployment(
                                dep_name, namespace
                            )

                            available = [
                                c.name for c in deployment.spec.template.spec.containers
                            ]
                            if len(available) != 1:
                                raise ValueError(
                                    "Set TGAT_CONTAINER for a multi-container Deployment"
                                )
                            containers = available

                        except Exception as exc:
                            raise RuntimeError(
                                f"Cannot select workload container: {exc}"
                            ) from exc

                    containers_patch = []

                    for container_name in containers:
                        resources: Dict[str, Dict[str, str]] = {
                            "requests": {},
                            "limits": {},
                        }

                        if desired_cpu:
                            resources["requests"]["cpu"] = desired_cpu

                            resources["limits"]["cpu"] = desired_cpu

                        if desired_mem:
                            resources["requests"]["memory"] = desired_mem

                            resources["limits"]["memory"] = desired_mem

                        containers_patch.append(
                            {"name": container_name, "resources": resources}
                        )

                    patch_body = {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": dep_name, "namespace": namespace},
                        "spec": {
                            "template": {"spec": {"containers": containers_patch}}
                        },
                    }

                    apps.patch_namespaced_deployment(
                        name=dep_name,
                        namespace=namespace,
                        body=patch_body,
                        field_manager=field_manager,
                    )
                    item_result["resources"] = {
                        "mode": "strategic-merge",
                        "containers": containers,
                    }

                except Exception as exc:
                    item_result["errors"].append("resources_patch_" "failed: " f"{exc}")

            # -----------------------------------------------------------------
            # Horizontal scaling
            # -----------------------------------------------------------------
            try:
                scale_body = {"spec": {"replicas": int(action.replicas)}}

                apps.patch_namespaced_deployment_scale(
                    name=dep_name, namespace=namespace, body=scale_body
                )

                item_result["replicas"] = int(action.replicas)

            except Exception as exc:
                item_result["errors"].append("scale_patch_failed: " f"{exc}")

            if not item_result["errors"]:
                report["patched"].append(dep_name)

            report["details"].append(item_result)

        return report
