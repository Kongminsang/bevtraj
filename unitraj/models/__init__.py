from .bevtp.BEVTP import BEVTP

__all__ = {
    'BEVTP': BEVTP,
}

def build_model(config):
    model = __all__[config.NAME](
        config=config
    )

    return model