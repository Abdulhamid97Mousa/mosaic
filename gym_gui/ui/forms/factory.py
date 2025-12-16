"""Worker form factory for training, evaluation, and policy selection UIs."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from qtpy import QtWidgets


TrainFormFactory = Callable[..., QtWidgets.QDialog]
PolicyFormFactory = Callable[..., QtWidgets.QDialog]
ResumeFormFactory = Callable[..., QtWidgets.QDialog]
EvaluationFormFactory = Callable[..., QtWidgets.QDialog]


class WorkerFormFactory:
    """Registry-backed factory for worker-specific UI forms.

    Supports:
    - Train forms: Configure training parameters
    - Policy forms: Load/select trained policies
    - Resume forms: Resume interrupted training
    - Evaluation forms: Configure policy evaluation
    """

    def __init__(self) -> None:
        self._train_forms: Dict[str, TrainFormFactory] = {}
        self._policy_forms: Dict[str, PolicyFormFactory] = {}
        self._resume_forms: Dict[str, ResumeFormFactory] = {}
        self._evaluation_forms: Dict[str, EvaluationFormFactory] = {}

    # ------------------------------------------------------------------
    def register_train_form(self, worker_id: str, factory: TrainFormFactory) -> None:
        if worker_id in self._train_forms:
            raise ValueError(f"Train form already registered for worker '{worker_id}'")
        self._train_forms[worker_id] = factory

    def register_policy_form(self, worker_id: str, factory: PolicyFormFactory) -> None:
        if worker_id in self._policy_forms:
            raise ValueError(f"Policy form already registered for worker '{worker_id}'")
        self._policy_forms[worker_id] = factory

    def register_resume_form(self, worker_id: str, factory: ResumeFormFactory) -> None:
        if worker_id in self._resume_forms:
            raise ValueError(f"Resume form already registered for worker '{worker_id}'")
        self._resume_forms[worker_id] = factory

    def register_evaluation_form(
        self, worker_id: str, factory: EvaluationFormFactory
    ) -> None:
        if worker_id in self._evaluation_forms:
            raise ValueError(
                f"Evaluation form already registered for worker '{worker_id}'"
            )
        self._evaluation_forms[worker_id] = factory

    # ------------------------------------------------------------------
    def create_train_form(self, worker_id: str, *args, **kwargs) -> QtWidgets.QDialog:
        factory = self._train_forms.get(worker_id)
        if factory is None:
            raise KeyError(f"No train form registered for worker '{worker_id}'")
        return factory(*args, **kwargs)

    def create_policy_form(self, worker_id: str, *args, **kwargs) -> QtWidgets.QDialog:
        factory = self._policy_forms.get(worker_id)
        if factory is None:
            raise KeyError(f"No policy form registered for worker '{worker_id}'")
        return factory(*args, **kwargs)

    def create_resume_form(self, worker_id: str, *args, **kwargs) -> QtWidgets.QDialog:
        factory = self._resume_forms.get(worker_id)
        if factory is None:
            raise KeyError(f"No resume form registered for worker '{worker_id}'")
        return factory(*args, **kwargs)

    def create_evaluation_form(
        self, worker_id: str, *args, **kwargs
    ) -> QtWidgets.QDialog:
        factory = self._evaluation_forms.get(worker_id)
        if factory is None:
            raise KeyError(f"No evaluation form registered for worker '{worker_id}'")
        return factory(*args, **kwargs)

    # ------------------------------------------------------------------
    def has_train_form(self, worker_id: str) -> bool:
        return worker_id in self._train_forms

    def has_policy_form(self, worker_id: str) -> bool:
        return worker_id in self._policy_forms

    def has_resume_form(self, worker_id: str) -> bool:
        return worker_id in self._resume_forms

    def has_evaluation_form(self, worker_id: str) -> bool:
        return worker_id in self._evaluation_forms


_factory = WorkerFormFactory()


def get_worker_form_factory() -> WorkerFormFactory:
    return _factory

