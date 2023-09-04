from pipelines import (
    Baseline,
    Mask2Formers_Swin_Base_RGB,
)

PIPELINE_FACTORY = {
        "baseline": Baseline,
        "mask2formers_swin_base_rgb": Mask2Formers_Swin_Base_RGB,
    }


def get_pipeline_component(
        config: dict,
    )-> dict:
    return PIPELINE_FACTORY[config['pipeline']](config).get_components()
