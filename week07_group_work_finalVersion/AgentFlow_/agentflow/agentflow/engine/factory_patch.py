# factory_patch.py
# Patches the AgentFlow engine factory to support Modal-hosted engines.
# Import this module once at your entry point (e.g. in __init__.py or before
# calling construct_solver) and the "modal" engine prefix will be available.
#
# Usage:
#   import agentflow.agentflow.engine.factory_patch  # noqa: F401

def _apply():
    # Try both common install layouts for agentflow
    factory_mod = None
    for mod_path in (
        "agentflow.agentflow.engine.factory",
        "agentflow.engine.factory",
    ):
        try:
            import importlib
            factory_mod = importlib.import_module(mod_path)
            break
        except ImportError:
            continue

    if factory_mod is None:
        raise ImportError(
            "Could not locate agentflow engine factory. "
            "Tried: agentflow.agentflow.engine.factory, agentflow.engine.factory"
        )

    _orig = factory_mod.create_llm_engine

    def _patched(model_string: str, temperature: float = 0.0, **kwargs):
        if model_string.startswith("modal"):
            from agentflow.agentflow.engine.modal_engine import ModalEngine
            return ModalEngine(model_id=model_string, temperature=temperature)
        return _orig(model_string, temperature=temperature, **kwargs)

    factory_mod.create_llm_engine = _patched


_apply()
