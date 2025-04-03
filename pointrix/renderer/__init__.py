from .dptr import RENDERER_REGISTRY

def parse_renderer(cfg, **kwargs):
    """
    Parse the renderer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    name = cfg.pop("name")
    if name == "DPTRRender":
        from .dptr import DPTRRender
        return DPTRRender(cfg, **kwargs)
    elif name == "DPTROrthoRender":
        from .dptr_ortho import DPTROrthoRender
        return DPTROrthoRender(cfg, **kwargs)
    elif name == "DPTROrthoEnhancedRender":
        from .dptr_ortho_enhanced import DPTROrthoEnhancedRender
        return DPTROrthoEnhancedRender(cfg, **kwargs)
    elif name == "DPTROrthoEnhancedRenderGabor":
        from .dptr_ortho_enhanced_gabor import DPTROrthoEnhancedRenderGabor
        return DPTROrthoEnhancedRenderGabor(cfg, **kwargs)
    else:
        raise ValueError(f"Renderer {name} not found")