"""Gradient handling module for MTL optimization.

This module provides classes for gradient manipulation, accumulation,
and conflict resolution in multi-task learning settings.
"""

import copy
from typing import Dict, Optional
import torch
from torch import nn
import random
import numpy as np

from media_bias_detection.utils.enums import AggregationMethod
from media_bias_detection.utils.logger import general_logger
from media_bias_detection.utils.common import rsetattr


class GradientError(Exception):
    """Custom exception for gradient-related errors."""
    pass


class GradsWrapper(nn.Module):
    """Wrapper for getting/setting gradients of trainable layers."""

    def __init__(self, *args, **kwargs):
        """Initialize GradsWrapper."""
        if type(self) == GradsWrapper:
            raise RuntimeError("Abstract class <GradsWrapper> must not be instantiated.")
        super().__init__()
        general_logger.debug("Initialized GradsWrapper")

    def get_grads(self) -> Dict[str, torch.Tensor]:
        """Get gradients of weights and biases for all trainable layers."""
        try:
            return {
                k: v.grad.clone() if v.grad is not None else None
                for k, v in dict(self.named_parameters()).items()
            }
        except Exception as e:
            raise GradientError(f"Failed to get gradients: {str(e)}")

    def set_grads(self, grads: Dict[str, torch.Tensor]) -> None:
        """Set gradients of weights and biases for all trainable layers."""
        try:
            for k, v in grads.items():
                rsetattr(self, f"{k}.grad", v)
        except Exception as e:
            raise GradientError(f"Failed to set gradients: {str(e)}")


class AccumulatorError(Exception):
    """Custom exception for accumulator-related errors."""
    pass


class Accumulator:
    """Abstract Accumulator for gradient handling."""

    def __init__(self):
        """Initialize abstract accumulator."""
        if type(self) == Accumulator:
            raise RuntimeError("Abstract class <Accumulator> must not be instantiated.")
        self.gradients = None
        self.n = 0

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Update gradient values (must be implemented by concrete classes)."""
        raise NotImplementedError

    def get_avg_gradients(self) -> Dict[str, torch.Tensor]:
        """Return gradients normalized across 0-axis."""
        try:
            if not self.gradients:
                raise AccumulatorError("No gradients available")

            out_gradients = copy.deepcopy(self.gradients)
            for k, v in self.gradients.items():
                out_gradients[k] /= self.n
                out_gradients[k] = out_gradients[k].squeeze(dim=0)
            return out_gradients
        except Exception as e:
            raise AccumulatorError(f"Failed to get average gradients: {str(e)}")

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Return raw gradients."""
        if not self.gradients:
            raise AccumulatorError("No gradients available")
        return self.gradients


class StackedAccumulator(Accumulator):
    """Accumulator that stacks gradients along 0-axis."""

    def __init__(self):
        """Initialize StackedAccumulator."""
        try:
            super().__init__()
            general_logger.debug("Initialized StackedAccumulator")
        except Exception as e:
            raise AccumulatorError(f"Failed to initialize StackedAccumulator: {str(e)}")

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Update by concatenating new gradients along 0-axis."""
        try:
            if not self.gradients:
                self.gradients = gradients
                # Unsqueeze all gradients for later concatenation
                for k, v in self.gradients.items():
                    self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
            else:
                for k, v in self.gradients.items():
                    new_value = gradients[k].unsqueeze(dim=0) * weight
                    self.gradients[k] = torch.cat((v, new_value), dim=0)
            self.n += 1
        except Exception as e:
            raise AccumulatorError(f"Failed to update stacked gradients: {str(e)}")

    def set_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Set gradients directly."""
        try:
            for k, v in self.gradients.items():
                self.gradients[k] = gradients[k].unsqueeze(dim=0)
        except Exception as e:
            raise AccumulatorError(f"Failed to set gradients: {str(e)}")


class RunningSumAccumulator(Accumulator):
    """Accumulator that maintains running sum of gradients."""

    def __init__(self):
        """Initialize RunningSumAccumulator."""
        try:
            super().__init__()
            general_logger.debug("Initialized RunningSumAccumulator")
        except Exception as e:
            raise AccumulatorError(f"Failed to initialize RunningSumAccumulator: {str(e)}")

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Update by summing gradients along 0-axis."""
        try:
            if not self.gradients:
                self.gradients = gradients
                # Unsqueeze all gradients for later addition
                for k, v in self.gradients.items():
                    self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
            else:
                for k, v in self.gradients.items():
                    new_value = gradients[k].unsqueeze(dim=0) * weight
                    self.gradients[k] = torch.add(v, new_value)
            self.n += 1
        except Exception as e:
            raise AccumulatorError(f"Failed to update running sum gradients: {str(e)}")


class GradientAggregator:
    """Aggregator class for combining possibly conflicting gradients into one 'optimal' grad."""

    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.MEAN):
        try:
            self.aggregation_method = aggregation_method
            self.accumulator = (
                RunningSumAccumulator() if aggregation_method == AggregationMethod.MEAN else StackedAccumulator()
            )
            self._conflicting_gradient_count = 0
            self._nonconflicting_gradient_count = 0
            general_logger.info(f"Initialized GradientAggregator with {aggregation_method}")
        except Exception as e:
            raise GradientError(f"Failed to initialize GradientAggregator: {str(e)}")

    def reset_accumulator(self) -> None:
        try:
            self.accumulator = (
                RunningSumAccumulator() if self.aggregation_method == AggregationMethod.MEAN else StackedAccumulator()
            )
            general_logger.debug("Reset gradient accumulator")
        except Exception as e:
            raise GradientError(f"Failed to reset accumulator: {str(e)}")

    def find_nonconflicting_grad(self, grad_tensor: torch.tensor) -> torch.tensor:
        try:
            if self.aggregation_method == AggregationMethod.PCGRAD:
                return self.pcgrad(grad_tensor).mean(dim=0)
            elif self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
                assert len(grad_tensor) == 2
                return self.pcgrad_online(grad_tensor)
            else:
                raise GradientError(f"Unsupported aggregation method: {self.aggregation_method}")
        except Exception as e:
            raise GradientError(f"Failed to find nonconflicting gradient: {str(e)}")

    def aggregate_gradients(self) -> Dict[str, torch.tensor]:
        try:
            conflicting_grads = self.accumulator.get_gradients()
            length = len(conflicting_grads[list(conflicting_grads.keys())[0]])

            if (self.aggregation_method == AggregationMethod.PCGRAD_ONLINE
                    or self.aggregation_method == AggregationMethod.MEAN):
                assert length == 1
                return self.accumulator.get_avg_gradients()

            elif self.aggregation_method == AggregationMethod.PCGRAD:
                conflicting_grads = [{k: v[i, ...] for k, v in conflicting_grads.items()} for i in range(length)]
                final_grad: Dict[str, torch.Tensor] = {}

                if len(conflicting_grads) == 1:
                    return conflicting_grads[0]

                keys = list(conflicting_grads[0].keys())
                for layer_key in keys:
                    list_of_st_grads = [st_grad[layer_key] for st_grad in conflicting_grads]
                    final_grad.update({layer_key: self.find_nonconflicting_grad(torch.stack(list_of_st_grads, dim=0))})

                return final_grad
            else:
                raise GradientError(f"Unsupported aggregation method: {self.aggregation_method}")
        except Exception as e:
            raise GradientError(f"Gradient aggregation failed: {str(e)}")

    def pcgrad(self, grad_tensor: torch.tensor) -> torch.tensor:
        try:
            pc_grads, num_of_tasks = grad_tensor.clone(), len(grad_tensor)
            original_shape = grad_tensor.shape

            pc_grads = pc_grads.view(num_of_tasks, -1)
            grad_tensor = grad_tensor.view(num_of_tasks, -1)

            for g_i in range(num_of_tasks):
                task_index = list(range(num_of_tasks))
                random.shuffle(task_index)
                for g_j in task_index:
                    dot_product = pc_grads[g_i].dot(grad_tensor[g_j])
                    if dot_product < 0:
                        pc_grads[g_i] -= (dot_product / (grad_tensor[g_j].norm() ** 2)) * grad_tensor[g_j]
                        self._conflicting_gradient_count += 1
                    else:
                        self._nonconflicting_gradient_count += 1
            return pc_grads.view(original_shape)
        except Exception as e:
            raise GradientError(f"PCGrad processing failed: {str(e)}")

    def pcgrad_online(self, grad_tensor: torch.tensor) -> torch.tensor:
        try:
            assert len(grad_tensor) == 2
            p = grad_tensor[0].view(-1)
            g = grad_tensor[-1].view(-1)

            dot_product = p.dot(g)
            if dot_product < 0:
                p = p - (dot_product / (g.norm() ** 2)) * g
                self._conflicting_gradient_count += 1
            else:
                self._nonconflicting_gradient_count += 1

            p += g
            return p.view(grad_tensor[0].shape)
        except Exception as e:
            raise GradientError(f"Online PCGrad processing failed: {str(e)}")

    def aggregate_gradients_online(self) -> Dict[str, torch.tensor]:
        try:
            conflicting_grads = self.accumulator.get_gradients()
            length = len(conflicting_grads[list(conflicting_grads.keys())[0]])
            conflicting_grads = [{k: v[i, ...] for k, v in conflicting_grads.items()} for i in range(length)]
            current_overall_grad: Dict[str, torch.Tensor] = {}

            if length == 1:
                return conflicting_grads[0]
            elif length == 2:
                keys = list(conflicting_grads[0].keys())
                for layer_key in keys:
                    list_of_st_grads = [st_grad[layer_key] for st_grad in conflicting_grads]
                    current_overall_grad.update(
                        {layer_key: self.find_nonconflicting_grad(torch.stack(list_of_st_grads, dim=0))}
                    )
                return current_overall_grad
            else:
                raise GradientError("Invalid gradient length for online aggregation")
        except Exception as e:
            raise GradientError(f"Online gradient aggregation failed: {str(e)}")

    def update(self, gradients: Dict[str, torch.tensor], scaling_weight: float) -> None:
        try:
            self.accumulator.update(gradients=gradients, weight=scaling_weight)
            if self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
                self.accumulator.set_gradients(gradients=self.aggregate_gradients_online())
        except Exception as e:
            raise GradientError(f"Failed to update gradients: {str(e)}")

    def get_conflicting_gradients_ratio(self) -> Optional[float]:
        try:
            if self.aggregation_method == AggregationMethod.MEAN:
                raise GradientError("Cannot get conflict ratio for MEAN method")
            if self._conflicting_gradient_count + self._nonconflicting_gradient_count == 0:
                raise GradientError("No gradients processed yet")
            return self._conflicting_gradient_count / (
                    self._conflicting_gradient_count + self._nonconflicting_gradient_count
            )
        except Exception as e:
            raise GradientError(f"Failed to calculate gradient conflict ratio: {str(e)}")