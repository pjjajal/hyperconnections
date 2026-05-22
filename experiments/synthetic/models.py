"""Model wrappers for synthetic tasks."""

import torch
import torch.nn as nn


class ZeroModule(nn.Module):
    """Zero output module - makes hyperconnection a pure stream dynamics test.

    With this module, the hyperconnection update becomes:
        H_{l+1} = A_l @ H_l + c_l * f(b_l^T @ H_l)^T
                = A_l @ H_l + 0
                = A_l @ H_l

    This isolates the effect of stream mixing from the nonlinear transformations.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


class IdentityModule(nn.Module):
    """Identity transformation - returns input unchanged."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class SimpleMLPModule(nn.Module):
    """Simple MLP for testing with learned transformations."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


class StreamDynamicsModel(nn.Module):
    """Wrapper model for testing stream dynamics on synthetic tasks.

    Uses a single hyperconnection layer applied n_layers times to test if
    one set of dynamics can be repeatedly applied to solve the task.

    For pure stream dynamics testing, use module_class=ZeroModule.
    """

    def __init__(
        self,
        n_streams: int,
        d: int,
        n_layers: int,
        hc_class,
        hc_kwargs: dict | None = None,
        module_class=ZeroModule,
        module_kwargs: dict | None = None,
    ):
        """
        Args:
            n_streams: Number of streams (n)
            d: Dimension of each stream
            n_layers: Number of times to apply the layer (L)
            hc_class: Hyperconnection class (e.g., ContinuousGenHyperConnections)
            hc_kwargs: Kwargs for hyperconnection initialization
            module_class: Module class for the nonlinear transformation (default: ZeroModule)
            module_kwargs: Kwargs for module initialization
        """
        super().__init__()
        self.n_streams = n_streams
        self.d = d
        self.n_layers = n_layers

        hc_kwargs = hc_kwargs or {}
        module_kwargs = module_kwargs or {}

        # For these synthetic tasks, we typically want m = n (no dimension reduction)
        m = hc_kwargs.get('m', n_streams)
        input_dim = n_streams * d
        embed_dim = m * d

        # Create a single layer that will be reused
        module = module_class(**module_kwargs) if module_kwargs else module_class()
        self.layer = hc_class(
            n=n_streams,
            m=m,
            input_dim=input_dim,
            embed_dim=embed_dim,
            module=module,
            **hc_kwargs,
        )

    def forward(self, H: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            H: Input stream state [B, n_streams, d]
            noise: Optional noise to add at specific layers [B, n_noise_layers, n_streams, d]

        Returns:
            Output stream state [B, n_streams, d]
        """
        B, n, d = H.shape
        assert n == self.n_streams and d == self.d

        # Flatten to [B, input_dim] for hyperconnection input
        x = H.reshape(B, -1)

        # Apply the same layer n_layers times
        noise_idx = 0
        for step in range(self.n_layers):
            x = self.layer(x)

            # Add noise if provided and within noise layer range
            if noise is not None and noise_idx < noise.shape[1]:
                x_reshaped = x.reshape(B, self.n_streams, self.d)
                x_reshaped = x_reshaped + noise[:, noise_idx]
                x = x_reshaped.reshape(B, -1)
                noise_idx += 1

        # Reshape back to [B, n_streams, d]
        return x.reshape(B, self.n_streams, self.d)
