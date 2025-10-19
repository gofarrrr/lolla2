# src/core/persistence/__init__.py
# Expose contracts and adapters for convenient import
from .contracts import IEventPersistence
from .adapters import SupabaseAdapter, FileAdapter
