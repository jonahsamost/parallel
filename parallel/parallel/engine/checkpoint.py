from __future__ import annotations

import copy
import json
import os
import random
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from ..state import Strategies


CHECKPOINT_FORMAT_VERSION = 1


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _cpu_tree(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


class CheckpointManager:
    """Coordinates portable full weights and exact same-topology checkpoints."""

    def __init__(self, engine):
        self.engine = engine

    @property
    def wrapper(self):
        model_parallel = getattr(self.engine, "mp_wrapper", None)
        if model_parallel is not None and model_parallel.is_active:
            return model_parallel
        return self.engine.fsdp_wrapper

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def _mesh_rank(self, dimension: Strategies) -> int:
        mesh = self.engine.pconfig.device_mesh
        if mesh is not None and dimension in getattr(mesh, "mesh_dim_names", ()):
            return mesh.get_local_rank(dimension)
        return 0

    @property
    def shard_rank(self) -> int:
        return self._mesh_rank(Strategies.DP_SHARD)

    @property
    def tp_rank(self) -> int:
        return self._mesh_rank(Strategies.TP)

    @property
    def model_parallel_active(self) -> bool:
        wrapper = getattr(self.engine, "mp_wrapper", None)
        return wrapper is not None and wrapper.is_active

    @property
    def is_model_writer(self) -> bool:
        return self._mesh_rank(Strategies.DP_REPLICATE) == 0

    def _broadcast_root_object(self, value: Any) -> Any:
        if not dist.is_initialized():
            return value
        payload = [value if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _collect_errors(self, local_error: Optional[str]) -> list[str]:
        if not dist.is_initialized():
            return [local_error] if local_error is not None else []
        errors: list[Optional[str]] = [None] * self.world_size
        dist.all_gather_object(errors, local_error)
        return [error for error in errors if error is not None]

    def _topology(self) -> dict[str, int]:
        pconfig = self.engine.pconfig
        return {
            "world_size": self.world_size,
            "dp_replicate_size": getattr(pconfig, "dp_replicate_size", 1),
            "dp_shard_size": getattr(pconfig, "dp_shard_size", 1),
            "tp_size": getattr(pconfig, "tp_size", 1),
            "ep_size": getattr(pconfig, "ep_size", 1),
            "expert_tp_size": getattr(pconfig, "expert_tp_size", 1),
        }

    def _resolved_config(self) -> Any:
        cfg = getattr(self.engine, "cfg", None)
        if cfg is None:
            return None
        try:
            return OmegaConf.to_container(cfg, resolve=True)
        except (TypeError, ValueError):
            return _jsonable(cfg)

    def _rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        device = self.engine.device
        if torch.device(device).type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(device)
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], self.engine.device)

    def full_state_dict(self):
        return self.wrapper.full_state_dict()

    def load_full_state_dict(self, state_dict, strict: bool = True):
        return self.wrapper.load_full_state_dict(state_dict, strict=strict)

    def sharded_state_dict(self) -> dict[str, Any]:
        return self.wrapper.sharded_state_dict()

    def load_sharded_state_dict(self, state_dict, strict: bool = True):
        return self.wrapper.load_sharded_state_dict(state_dict, strict=strict)

    def save_full_model(self, path: str | Path) -> None:
        """Save a portable CPU model state dict. Every rank must call this."""
        path = Path(path)
        state = self.full_state_dict()
        root_result = None
        if self.rank == 0:
            temporary_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
            try:
                if path.exists():
                    raise FileExistsError(f"Checkpoint already exists: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state, temporary_path)
                os.replace(temporary_path, path)
                root_result = {"error": None}
            except Exception as error:
                if temporary_path.exists():
                    temporary_path.unlink()
                root_result = {"error": f"{type(error).__name__}: {error}"}
        root_result = self._broadcast_root_object(root_result)
        if root_result["error"] is not None:
            raise RuntimeError(f"Failed to save full model: {root_result['error']}")

    def load_full_model(self, path: str | Path, strict: bool = True):
        """Load a portable CPU model state dict from rank zero."""
        state = None
        root_result = None
        if self.rank == 0:
            try:
                state = torch.load(Path(path), map_location="cpu", weights_only=False)
                root_result = {"error": None}
            except Exception as error:
                root_result = {"error": f"{type(error).__name__}: {error}"}
        root_result = self._broadcast_root_object(root_result)
        if root_result["error"] is not None:
            raise RuntimeError(f"Failed to load full model: {root_result['error']}")
        return self.load_full_state_dict(state, strict=strict)

    def _prepare_checkpoint_directory(self, path: Path) -> Path:
        result = None
        if self.rank == 0:
            try:
                if path.exists():
                    raise FileExistsError(f"Checkpoint already exists: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
                temporary.mkdir()
                result = {"path": str(temporary), "error": None}
            except Exception as error:
                result = {
                    "path": None,
                    "error": f"{type(error).__name__}: {error}",
                }
        result = self._broadcast_root_object(result)
        if result["error"] is not None:
            raise RuntimeError(f"Failed to initialize checkpoint: {result['error']}")
        return Path(result["path"])

    def _model_file_name(self, shard_rank: int, tp_rank: int | None = None) -> str:
        if self.model_parallel_active:
            tp_rank = self.tp_rank if tp_rank is None else tp_rank
            return f"model-tp-{tp_rank:05d}-dp-shard-{shard_rank:05d}.pt"
        return f"model-shard-{shard_rank:05d}.pt"

    def _model_file_names(self) -> list[str]:
        topology = self._topology()
        if self.model_parallel_active:
            return [
                self._model_file_name(dp_shard_rank, tp_rank)
                for tp_rank in range(topology["tp_size"])
                for dp_shard_rank in range(topology["dp_shard_size"])
            ]
        return [
            self._model_file_name(shard_rank)
            for shard_rank in range(topology["dp_shard_size"])
        ]

    def _rank_file_name(self, rank: int) -> str:
        return f"rank-state-{rank:05d}.pt"

    def save(
        self,
        path: str | Path,
        *,
        step: int,
        dataloader_state: Optional[dict[str, Any]] = None,
        eval_dataloader_state: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Atomically save an exact-resume, same-topology sharded checkpoint."""
        path = Path(path)
        temporary = self._prepare_checkpoint_directory(path)
        local_error = None
        try:
            if self.is_model_writer:
                model_payload = {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "model": self.sharded_state_dict(),
                    "optimizer": _cpu_tree(self.engine.optimizer.state_dict()),
                }
                torch.save(
                    model_payload,
                    temporary / self._model_file_name(
                        self.shard_rank, self.tp_rank
                    ),
                )

            scaler = getattr(self.engine, "grad_scaler", None)
            scheduler = getattr(self.engine, "scheduler", None)
            rank_payload = {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "rank": self.rank,
                "rng": self._rng_state(),
                "dataloader": copy.deepcopy(dataloader_state),
                "eval_dataloader": copy.deepcopy(eval_dataloader_state),
                "grad_scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(rank_payload, temporary / self._rank_file_name(self.rank))
        except Exception as error:
            local_error = f"rank {self.rank}: {type(error).__name__}: {error}"

        errors = self._collect_errors(local_error)
        if errors:
            if self.rank == 0:
                shutil.rmtree(temporary, ignore_errors=True)
            raise RuntimeError("Failed to write checkpoint: " + " | ".join(errors))

        finalize_result = None
        if self.rank == 0:
            try:
                dp_shard_size = self._topology()["dp_shard_size"]
                manifest = {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "kind": "sharded-training-state",
                    "step": int(step),
                    "topology": self._topology(),
                    "layout": self.wrapper.checkpoint_layout(),
                    "model_files": self._model_file_names(),
                    "rank_files": [
                        self._rank_file_name(rank) for rank in range(self.world_size)
                    ],
                    "config": self._resolved_config(),
                    "metadata": _jsonable(metadata or {}),
                }
                manifest_path = temporary / "manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
                os.replace(temporary, path)
                finalize_result = {"error": None}
            except Exception as error:
                shutil.rmtree(temporary, ignore_errors=True)
                finalize_result = {
                    "error": f"{type(error).__name__}: {error}",
                }
        finalize_result = self._broadcast_root_object(finalize_result)
        if finalize_result["error"] is not None:
            raise RuntimeError(
                f"Failed to finalize checkpoint: {finalize_result['error']}"
            )

    def _read_manifest(self, path: Path) -> dict[str, Any]:
        result = None
        if self.rank == 0:
            try:
                if not (path / "COMPLETE").is_file():
                    raise RuntimeError("Checkpoint has no COMPLETE marker")
                manifest = json.loads(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
                result = {"manifest": manifest, "error": None}
            except Exception as error:
                result = {
                    "manifest": None,
                    "error": f"{type(error).__name__}: {error}",
                }
        result = self._broadcast_root_object(result)
        if result["error"] is not None:
            raise RuntimeError(f"Failed to read checkpoint: {result['error']}")
        return result["manifest"]

    def load(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Restore model, optimizer, scaler, scheduler, RNG, and data position."""
        path = Path(path)
        manifest = self._read_manifest(path)
        if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported checkpoint version: {manifest.get('format_version')}"
            )
        if manifest.get("kind") != "sharded-training-state":
            raise RuntimeError(f"Unsupported checkpoint kind: {manifest.get('kind')}")
        if manifest.get("topology") != self._topology():
            raise RuntimeError(
                "Checkpoint topology does not match the current distributed topology: "
                f"saved={manifest.get('topology')}, current={self._topology()}"
            )
        if manifest.get("layout") != self.wrapper.checkpoint_layout():
            raise RuntimeError("Checkpoint model layout does not match the current model")
        expected_model_files = self._model_file_names()
        expected_rank_files = [
            self._rank_file_name(rank) for rank in range(self.world_size)
        ]
        if manifest.get("model_files") != expected_model_files:
            raise RuntimeError("Checkpoint manifest has an invalid model file set")
        if manifest.get("rank_files") != expected_rank_files:
            raise RuntimeError("Checkpoint manifest has an invalid rank-state file set")

        local_error = None
        rank_payload = None
        try:
            model_payload = torch.load(
                path / self._model_file_name(self.shard_rank, self.tp_rank),
                map_location="cpu",
                weights_only=False,
            )
            if model_payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise RuntimeError("Model shard has an unsupported format version")
            rank_payload = torch.load(
                path / self._rank_file_name(self.rank),
                map_location="cpu",
                weights_only=False,
            )
            if rank_payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise RuntimeError("Rank state has an unsupported format version")
            if rank_payload.get("rank") != self.rank:
                raise RuntimeError(
                    f"Loaded rank state for {rank_payload.get('rank')} on rank {self.rank}"
                )
            rng_state = rank_payload.get("rng")
            required_rng_keys = {"python", "numpy", "torch"}
            if not isinstance(rng_state, dict) or not required_rng_keys.issubset(rng_state):
                raise RuntimeError("Rank state has incomplete RNG state")
            if torch.device(self.engine.device).type == "cuda" and "cuda" not in rng_state:
                raise RuntimeError("Rank state has no CUDA RNG state")

            scaler = getattr(self.engine, "grad_scaler", None)
            scheduler = getattr(self.engine, "scheduler", None)
            saved_scaler = rank_payload.get("grad_scaler")
            saved_scheduler = rank_payload.get("scheduler")
            if (scaler is None) != (saved_scaler is None):
                raise RuntimeError("Checkpoint grad-scaler configuration does not match")
            if (scheduler is None) != (saved_scheduler is None):
                raise RuntimeError("Checkpoint scheduler configuration does not match")

            self.load_sharded_state_dict(model_payload["model"], strict=strict)
            self.engine.optimizer.load_state_dict(model_payload["optimizer"])
            if scaler is not None:
                scaler.load_state_dict(saved_scaler)
            if scheduler is not None:
                scheduler.load_state_dict(saved_scheduler)
        except Exception as error:
            local_error = f"rank {self.rank}: {type(error).__name__}: {error}"

        errors = self._collect_errors(local_error)
        if errors:
            raise RuntimeError("Failed to load checkpoint: " + " | ".join(errors))

        self._restore_rng_state(rank_payload["rng"])
        return {
            "step": manifest["step"],
            "dataloader_state": rank_payload.get("dataloader"),
            "eval_dataloader_state": rank_payload.get("eval_dataloader"),
            "metadata": manifest.get("metadata", {}),
            "config": manifest.get("config"),
        }
