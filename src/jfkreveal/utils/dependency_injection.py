"""
Dependency injection utilities for JFKReveal.

This module provides a lightweight dependency injection framework to make
external dependencies explicit, simplify testing, and improve modularity.
"""
from typing import Dict, Any, Type, TypeVar, Generic, Optional, Protocol, Callable, cast
import inspect
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

T = TypeVar('T')

class Provider(Generic[T], Protocol):
    """Protocol defining a provider interface."""
    def __call__(self) -> T:
        """Provide an instance of T."""
        ...

class DependencyContainer:
    """
    A simple dependency container that manages dependency registrations and resolutions.
    
    This container handles singleton and factory registrations, allowing for flexible
    dependency management without requiring a full-fledged DI framework.
    """
    
    def __init__(self):
        """Initialize an empty dependency container."""
        self._registrations: Dict[Type, Provider] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register_singleton(self, interface_type: Type[T], instance: T) -> None:
        """
        Register an existing instance as a singleton for the given interface type.
        
        Args:
            interface_type: The type to register the singleton for
            instance: The singleton instance
        """
        self._singletons[interface_type] = instance
        logger.debug(f"Registered singleton for {interface_type.__name__}")
    
    def register_singleton_factory(self, interface_type: Type[T], factory: Provider[T]) -> None:
        """
        Register a factory function that will create a singleton instance on first resolution.
        
        Args:
            interface_type: The type to register the singleton factory for
            factory: A callable that creates an instance of the interface type
        """
        def singleton_provider() -> T:
            if interface_type not in self._singletons:
                self._singletons[interface_type] = factory()
                logger.debug(f"Created singleton from factory for {interface_type.__name__}")
            return self._singletons[interface_type]
        
        self._registrations[interface_type] = singleton_provider
        logger.debug(f"Registered singleton factory for {interface_type.__name__}")
    
    def register_factory(self, interface_type: Type[T], factory: Provider[T]) -> None:
        """
        Register a factory function that will create a new instance on each resolution.
        
        Args:
            interface_type: The type to register the factory for
            factory: A callable that creates an instance of the interface type
        """
        self._registrations[interface_type] = factory
        logger.debug(f"Registered factory for {interface_type.__name__}")
    
    def resolve(self, interface_type: Type[T]) -> T:
        """
        Resolve a dependency for the given interface type.
        
        Args:
            interface_type: The type to resolve
            
        Returns:
            An instance of the requested type
            
        Raises:
            KeyError: If the interface type is not registered
        """
        # Check if there's a singleton instance first
        if interface_type in self._singletons:
            return self._singletons[interface_type]
        
        # Check if there's a factory or singleton factory
        if interface_type in self._registrations:
            return self._registrations[interface_type]()
        
        # If not found, raise an error
        raise KeyError(f"No registration found for {interface_type.__name__}")
    
    def build_instance(self, cls: Type[T], **kwargs) -> T:
        """
        Build an instance of the given class by resolving its dependencies from the container.
        
        Args:
            cls: The class to instantiate
            **kwargs: Additional constructor arguments to override
            
        Returns:
            An instance of the class with dependencies resolved
            
        Raises:
            KeyError: If any required dependency is not registered
        """
        # Get constructor parameters
        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters
        
        # Skip 'self' parameter
        parameters = {name: param for name, param in parameters.items() if name != 'self'}
        
        # Prepare constructor arguments
        args = {}
        
        for name, param in parameters.items():
            # Skip if already provided in kwargs
            if name in kwargs:
                continue
            
            # Handle parameters with and without type annotations
            if param.annotation != inspect.Parameter.empty:
                # Handle Optional types
                origin = getattr(param.annotation, "__origin__", None)
                args_attr = getattr(param.annotation, "__args__", None)
                
                # Check if it's Optional[Type]
                if origin is Optional and args_attr:
                    # Get the inner type from __args__
                    inner_type = param.annotation.__args__[0]
                    
                    # Try to resolve the inner type, even if it has a default value
                    try:
                        args[name] = self.resolve(inner_type)
                        continue  # Successfully resolved, continue to next parameter
                    except KeyError:
                        # If inner type can't be resolved, continue to default value check
                        pass
                else:
                    # Regular type (not Optional), try to resolve if it doesn't have a default
                    if param.default == inspect.Parameter.empty:  # No default value
                        try:
                            args[name] = self.resolve(param.annotation)
                            continue  # Successfully resolved, continue to next parameter
                        except KeyError:
                            # No registration found and no default, so raise an error
                            raise KeyError(f"Cannot resolve parameter '{name}' of type {param.annotation} for {cls.__name__}")
            
            # If we reach here, either the parameter has a default value,
            # or it's an Optional type that couldn't be resolved,
            # or it has no type annotation
            if param.default == inspect.Parameter.empty:
                # No default value and no resolution possible
                raise KeyError(f"Cannot resolve parameter '{name}' for {cls.__name__}")
            
            # Otherwise, let the default value be used by not setting it in args
            
        # Create instance with resolved arguments and overrides
        instance = cls(**{**args, **kwargs})
        return instance


# Default global container
_default_container = DependencyContainer()

def get_container() -> DependencyContainer:
    """Get the default dependency container."""
    return _default_container

def register_singleton(interface_type: Type[T], instance: T) -> None:
    """Register a singleton instance with the default container."""
    _default_container.register_singleton(interface_type, instance)

def register_singleton_factory(interface_type: Type[T], factory: Provider[T]) -> None:
    """Register a singleton factory with the default container."""
    _default_container.register_singleton_factory(interface_type, factory)

def register_factory(interface_type: Type[T], factory: Provider[T]) -> None:
    """Register a factory with the default container."""
    _default_container.register_factory(interface_type, factory)

def resolve(interface_type: Type[T]) -> T:
    """Resolve a dependency from the default container."""
    return _default_container.resolve(interface_type)

def build_instance(cls: Type[T], **kwargs) -> T:
    """Build an instance using the default container to resolve dependencies."""
    return _default_container.build_instance(cls, **kwargs)