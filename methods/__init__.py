from .source_only import run_iter_source_only
from .lirr import run_iter_lirr
train_method = {
    'source_only':run_iter_source_only,
    'lirr': run_iter_lirr
}