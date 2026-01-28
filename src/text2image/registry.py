"""Registry for text-to-image backends.

This module provides a factory pattern for easily creating and switching between
different text-to-image backends.
"""

from typing import Dict, Any, List
from .base import Text2ImageBackend
from .stable_diffusion import StableDiffusionBackend
from .dalle import DalleBackend


# Registry mapping backend names to classes TODO: confirm aliases
T2I_BACKENDS = {
    "stable_diffusion": StableDiffusionBackend,
    "sdxl": StableDiffusionBackend,  # Alias
    "sd": StableDiffusionBackend,     # Alias
    "dalle": DalleBackend,
    "dall-e": DalleBackend,           # Alias
    "dalle3": DalleBackend,           # Alias
}


def get_t2i_backend(backend_name: str, config: Dict[str, Any]) -> Text2ImageBackend:
    """Factory function to create a text-to-image backend.

    Args:
        backend_name: Name of the backend (e.g., "stable_diffusion", "dalle")
        config: Configuration dictionary for the backend

    Returns:
        Initialized text-to-image backend

    Raises:
        ValueError: If backend_name is not recognized

    Example:
        >>> config = {
        ...     'model_id': 'stabilityai/stable-diffusion-xl-base-1.0',
        ...     'device': 'cuda',
        ...     'guidance_scale': 7.5
        ... }
        >>> backend = get_t2i_backend('stable_diffusion', config)
        >>> images = backend.generate(['a photo of a cat'])
    """
    backend_name = backend_name.lower()

    if backend_name not in T2I_BACKENDS:
        available = ", ".join(list_t2i_backends())
        raise ValueError(
            f"Unknown T2I backend: '{backend_name}'. "
            f"Available backends: {available}"
        )

    backend_class = T2I_BACKENDS[backend_name]
    return backend_class(config)


def list_t2i_backends() -> List[str]:
    """List all available text-to-image backend names.

    Returns:
        List of backend names (without aliases)
    """
    # Return unique backend names (without aliases)
    seen = set()
    unique_backends = []
    for name, cls in T2I_BACKENDS.items():
        if cls not in seen:
            unique_backends.append(name)
            seen.add(cls)
    return unique_backends


def register_t2i_backend(name: str, backend_class: type):
    """Register a new text-to-image backend.

    This allows users to add custom backends without modifying the registry.

    Args:
        name: Name to register the backend under
        backend_class: Backend class (must inherit from Text2ImageBackend)

    Raises:
        TypeError: If backend_class doesn't inherit from Text2ImageBackend

    Example:
        >>> class MyCustomBackend(Text2ImageBackend):
        ...     # Implementation
        ...     pass
        >>> register_t2i_backend('my_backend', MyCustomBackend)
    """
    if not issubclass(backend_class, Text2ImageBackend):
        raise TypeError(
            f"Backend class must inherit from Text2ImageBackend, "
            f"got {backend_class.__name__}"
        )

    T2I_BACKENDS[name.lower()] = backend_class
    print(f"Registered T2I backend: '{name}'")
