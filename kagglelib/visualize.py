import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from typing import List

def get_uniques(input_df: pd.DataFrame, column):
    s = input_df[column]
    return set(s.dropna().unique())


def plot_intersection(
    left: pd.DataFrame, 
    right: pd.DataFrame, 
    target_column: str, 
    ax: plt.Axes = None, 
    set_labels: List[str]=None
):
    venn2(
        subsets=(get_uniques(left, target_column), get_uniques(right, target_column)),
        set_labels=set_labels or ("Train", "Test"),
        ax=ax
    )
    ax.set_title(target_column)