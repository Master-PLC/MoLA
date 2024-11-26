from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_finetune_group import \
    Exp_Long_Term_Forecast_Finetune_Group
from .exp_long_term_forecasting_iterative import \
    Exp_Long_Term_Forecast_Iterative

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'long_term_forecast_iterative': Exp_Long_Term_Forecast_Iterative,
    'long_term_forecast_finetune_group': Exp_Long_Term_Forecast_Finetune_Group,
}