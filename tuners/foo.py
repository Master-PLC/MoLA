from .adapter import PatchTSTAdapter, TransformerAdapter, iTransformerAdapter
from .fourier import iTransformerFourier
from .ia3 import iTransformerIA3
from .lora import (AutoformerLora, FEDformerLora, InformerLora, PatchTSTLora,
                   TransformerLora, iTransformerLora, iTransformerLoraHeader,
                   iTransformerLoraHeaderFixed, iTransformerMultiLora)
from .molora import iTransformerMoLora, TransformerMoLora, InformerMoLora, AutoformerMoLora
from .moia3 import iTransformerMoIA3
from .moadapter import iTransformerMoAdapter


TUNER_DICT = {
    'PatchTSTAdapter': PatchTSTAdapter,
    'TransformerAdapter': TransformerAdapter,
    'iTransformerAdapter': iTransformerAdapter,
    'iTransformerFourier': iTransformerFourier,
    'iTransformerIA3': iTransformerIA3,
    'PatchTSTLora': PatchTSTLora,
    'TransformerLora': TransformerLora,
    'iTransformerLora': iTransformerLora,
    'iTransformerMultiLora': iTransformerMultiLora,
    'AutoformerLora': AutoformerLora,
    'FEDformerLora': FEDformerLora,
    'iTransformerMoLora': iTransformerMoLora,
    'iTransformerLoraHeader': iTransformerLoraHeader,
    'iTransformerLoraHeaderFixed': iTransformerLoraHeaderFixed,
    'InformerLora': InformerLora,
    'TransformerMoLora': TransformerMoLora,
    'InformerMoLora': InformerMoLora,
    'AutoformerMoLora': AutoformerMoLora,
    'iTransformerMoIA3': iTransformerMoIA3,
    'iTransformerMoAdapter': iTransformerMoAdapter
}


PARAM_TARGET_DICT = {
    'PatchTSTAdapter': {'param_target': 'adapter'},
    'TransformerAdapter': {'param_target': 'adapter'},
    'iTransformerAdapter': {'param_target': 'adapter'},
    'iTransformerFourier': {'param_target': 'spectrum'},
    'iTransformerIA3': {'param_target': 'ia3'},
    'PatchTSTLora': {'param_target': 'lora'},
    'TransformerLora': {'param_target': 'lora'},
    'iTransformerLora': {'param_target': 'lora'},
    'iTransformerMultiLora': {'param_target': 'lora'},
    'AutoformerLora': {'param_target': 'lora'},
    'FEDformerLora': {'param_target': 'lora'},
    'iTransformerMoLora': {'param_target': 'lora_score', 'param_other': 'lora'},
    'iTransformerLoraHeader': {'param_target': 'lora'},
    'iTransformerLoraHeaderFixed': {'param_target': 'lora'},
    'InformerLora': {'param_target': 'lora'},
    'TransformerMoLora': {'param_target': 'lora_score', 'param_other': 'lora'},
    'InformerMoLora': {'param_target': 'lora_score', 'param_other': 'lora'},
    'AutoformerMoLora': {'param_target': 'lora_score', 'param_other': 'lora'},
    'iTransformerMoIA3': {'param_target': 'ia3_*_score', 'param_other': 'ia3'},
    'iTransformerMoAdapter': {'param_target': 'adapter_score', 'param_other': 'adapter'}
}