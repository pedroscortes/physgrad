"""
JAX integration for PhysGrad.
"""

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:
    try:
        from . import physgrad_cpp

        class JAXSimulation:
            """JAX-based simulation wrapper."""
            def __init__(self, config):
                self.config = config

        def register_jax_ops():
            """Register JAX XLA operations."""
            if hasattr(physgrad_cpp, 'register_jax_primitives'):
                physgrad_cpp.register_jax_primitives()

        def array_to_physgrad(array):
            """Convert JAX array to PhysGrad format."""
            return jnp.asarray(array)

        def physgrad_to_array(data):
            """Convert PhysGrad data to JAX array."""
            return jnp.array(data)

    except ImportError:
        JAXSimulation = None
        register_jax_ops = None
        array_to_physgrad = None
        physgrad_to_array = None
else:
    JAXSimulation = None
    register_jax_ops = None
    array_to_physgrad = None
    physgrad_to_array = None