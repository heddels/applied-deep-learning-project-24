"""Gradient handling module for MTL optimization.

This module provides classes for gradient manipulation, accumulation,
and conflict resolution in multi-task learning settings.
"""

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
    """Base class for gradient manipulation.

    This class provides functionality to get and set gradients
    across model components.
    """

    def __init__(self):
        if type(self) == GradsWrapper:
            raise RuntimeError("Abstract class <GradsWrapper> must not be instantiated.")
        super().__init__()

    def get_grads(self) -> Dict[str, torch.Tensor]:
        """Get gradients from all parameters.

        Returns:
            Dictionary mapping parameter names to their gradients
        """
        return {
            k: v.grad.clone() if v.grad is not None else None
            for k, v in dict(self.named_parameters()).items()
        }

    def set_grads(self, grads: Dict[str, torch.Tensor]) -> None:
        """Set gradients for all parameters.

        Args:
            grads: Dictionary mapping parameter names to gradients
        """
        for k, v in grads.items():
            rsetattr(self, f"{k}.grad", v)


class Accumulator:
    """Base class for gradient accumulation.

    This class defines the interface for different gradient
    accumulation strategies.
    """

    def __init__(self):
        if type(self) == Accumulator:
            raise RuntimeError("Abstract class <Accumulator> must not be instantiated.")
        self.gradients: Optional[Dict[str, torch.Tensor]] = None
        self.n: int = 0

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Update accumulated gradients.

        Args:
            gradients: New gradients to accumulate
            weight: Weight for the new gradients
        """
        raise NotImplementedError

    def get_avg_gradients(self) -> Dict[str, torch.Tensor]:
        """Get normalized accumulated gradients.

        Returns:
            Normalized gradients
        """
        if not self.gradients:
            raise GradientError("No gradients accumulated")

        out_gradients = {}
        for k, v in self.gradients.items():
            if self.n > 0:
                out_gradients[k] = v / self.n
            else:
                out_gradients[k] = v
            out_gradients[k] = out_gradients[k].squeeze(dim=0)

        return out_gradients


class StackedAccumulator(Accumulator):
    """Accumulator that stacks gradients along first dimension."""

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Stack new gradients with existing ones.

        Args:
            gradients: New gradients to stack
            weight: Weight for the new gradients
        """
        try:
            if not self.gradients:
                self.gradients = gradients
                for k, v in self.gradients.items():
                    self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
            else:
                for k, v in self.gradients.items():
                    new_value = gradients[k].unsqueeze(dim=0) * weight
                    self.gradients[k] = torch.cat((v, new_value), dim=0)
            self.n += 1

        except Exception as e:
            raise GradientError(f"Failed to update stacked gradients: {str(e)}")

    def set_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Set gradients directly.

        Args:
            gradients: Gradients to set
        """
        for k, v in self.gradients.items():
            self.gradients[k] = gradients[k].unsqueeze(dim=0)


class RunningSumAccumulator(Accumulator):
    """Accumulator that maintains running sum of gradients."""

    def update(self, gradients: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
        """Add new gradients to running sum.

        Args:
            gradients: New gradients to add
            weight: Weight for the new gradients
        """
        try:
            if not self.gradients:
                self.gradients = gradients
                for k, v in self.gradients.items():
                    self.gradients[k] = self.gradients[k].unsqueeze(dim=0) * weight
            else:
                for k, v in self.gradients.items():
                    new_value = gradients[k].unsqueeze(dim=0) * weight
                    self.gradients[k] = torch.add(v, new_value)
            self.n += 1

        except Exception as e:
            raise GradientError(f"Failed to update running sum gradients: {str(e)}")


class GradientAggregator:
    """Handles aggregation of gradients from multiple tasks.

    This class implements different methods for combining potentially
    conflicting gradients in multi-task learning.

    Attributes:
        aggregation_method: Method to use for gradient aggregation
        accumulator: Gradient accumulator instance
        _conflicting_gradient_count: Count of conflicting gradients
        _nonconflicting_gradient_count: Count of non-conflicting gradients
    """

    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.MEAN):
        self.aggregation_method = aggregation_method
        self.accumulator = (
            RunningSumAccumulator() if aggregation_method == AggregationMethod.MEAN
            else StackedAccumulator()
        )
        self._conflicting_gradient_count = 0
        self._nonconflicting_gradient_count = 0

    def reset_accumulator(self) -> None:
        """Reset the gradient accumulator."""
        self.accumulator = (
            RunningSumAccumulator() if self.aggregation_method == AggregationMethod.MEAN
            else StackedAccumulator()
        )

    def find_nonconflicting_grad(self, grad_tensor: torch.Tensor) -> torch.Tensor:
        """Find non-conflicting gradient using specified method.

        Args:
            grad_tensor: Tensor of gradients to process

        Returns:
            Processed non-conflicting gradient
        """
        try:
            if self.aggregation_method == AggregationMethod.PCGRAD:
                return self.pcgrad(grad_tensor).mean(dim=0)
            elif self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
                assert len(grad_tensor) == 2
                return self.pcgrad_online(grad_tensor)
            else:
                raise GradientError(f"Unsupported aggregation method: {self.aggregation_method}")

        except Exception as e:
            raise GradientError(f"Failed to find non-conflicting gradient: {str(e)}")

    def pcgrad(self, grad_tensor: torch.Tensor) -> torch.Tensor:
        """Project conflicting gradients using PCGrad algorithm.

        Args:
            grad_tensor: Tensor of gradients

        Returns:
            Processed gradients
        """
        try:
            pc_grads = grad_tensor.clone()
            num_tasks = len(grad_tensor)
            original_shape = grad_tensor.shape

            pc_grads = pc_grads.view(num_tasks, -1)
            grad_tensor = grad_tensor.view(num_tasks, -1)

            for g_i in range(num_tasks):
                task_index = list(range(num_tasks))
                random.shuffle(task_index)
                for g_j in task_index:
                    dot_product = pc_grads[g_i].dot(grad_tensor[g_j])
                    if dot_product < 0:
                        pc_grads[g_i] -= (
                                (dot_product / (grad_tensor[g_j].norm() ** 2))
                                * grad_tensor[g_j]
                        )
                        self._conflicting_gradient_count += 1
                    else:
                        self._nonconflicting_gradient_count += 1

            return pc_grads.view(original_shape)

        except Exception as e:
            raise GradientError(f"PCGrad processing failed: {str(e)}")

    def pcgrad_online(self, grad_tensor: torch.Tensor) -> torch.Tensor:
        """Apply PCGrad in online setting.

        Args:
            grad_tensor: Tensor of gradients (must be length 2)

        Returns:
            Processed gradient
        """
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

    def aggregate_gradients(self) -> Dict[str, torch.Tensor]:
        """Aggregate accumulated gradients.

        Returns:
            Aggregated gradients
        """
        try:
            conflicting_grads = self.accumulator.get_gradients()
            if not conflicting_grads:
                raise GradientError("No gradients to aggregate")

            length = len(conflicting_grads[list(conflicting_grads.keys())[0]])

            if (self.aggregation_method == AggregationMethod.PCGRAD_ONLINE
                    or self.aggregation_method == AggregationMethod.MEAN):
                assert length == 1
                return self.accumulator.get_avg_gradients()

            elif self.aggregation_method == AggregationMethod.PCGRAD:
                if length == 1:
                    return conflicting_grads

                conflicting_grads = [
                    {k: v[i, ...] for k, v in conflicting_grads.items()}
                    for i in range(length)
                ]

                final_grad = {}
                keys = list(conflicting_grads[0].keys())

                for layer_key in keys:
                    grad_list = [st_grad[layer_key] for st_grad in conflicting_grads]
                    final_grad[layer_key] = self.find_nonconflicting_grad(
                        torch.stack(grad_list, dim=0)
                    )

                return final_grad

            else:
                raise GradientError(f"Unsupported aggregation method: {self.aggregation_method}")

        except Exception as e:
            raise GradientError(f"Gradient aggregation failed: {str(e)}")

    def update(self, gradients: Dict[str, torch.Tensor], scaling_weight: float) -> None:
        """Update accumulated gradients.

        Args:
            gradients: New gradients to accumulate
            scaling_weight: Weight for the new gradients
        """
        try:
            self.accumulator.update(gradients=gradients, weight=scaling_weight)

            if self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
                self.accumulator.set_gradients(
                    gradients=self.aggregate_gradients_online()
                )

        except Exception as e:
            raise GradientError(f"Failed to update gradients: {str(e)}")

    def get_conflicting_gradients_ratio(self) -> Optional[float]:
        """Get ratio of conflicting to total gradients.

        Returns:
            Ratio of conflicting gradients, or None if no gradients processed
        """
        total = self._conflicting_gradient_count + self._nonconflicting_gradient_count
        if total == 0:
            return None
        return self._conflicting_gradient_count / total