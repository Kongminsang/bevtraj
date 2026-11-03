from .bevtraj.BEVTraj import BEVTraj


__all__ = {
    'BEVTraj': BEVTraj,
}

def build_model(config):
    model = __all__[config.NAME](
        config=config
    )

    return model
