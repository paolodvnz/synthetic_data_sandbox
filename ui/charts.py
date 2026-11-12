"""Altair charts of the Validate step: the two association heatmaps and the histogram of one column."""

from __future__ import annotations

import altair as alt
import pandas as pd

ASSOCIATION_DOMAIN = [0.0, 1.0]         # every association is in [0, 1]: one scale for both heatmaps
HEATMAP_SCHEME = "blues"
CELL_PIXELS = 22                        # side of a heatmap cell
MIN_HEATMAP_PIXELS = 160
HISTOGRAM_HEIGHT = 280
SOURCE_COLORS = {"Real": "#7f8c9d", "Synthetic": "#e4572e"}


def heatmap(frame: pd.DataFrame, columns: list[str], title: str) -> alt.Chart:
    """One association matrix; the colour scale is fixed, so two heatmaps compare by colour."""
    side = max(MIN_HEATMAP_PIXELS, CELL_PIXELS * len(columns))
    return alt.Chart(frame, title=title).mark_rect().encode(
        x=alt.X("column:N", sort=columns, title=None),
        y=alt.Y("row:N", sort=columns, title=None),
        color=alt.Color("association:Q", scale=alt.Scale(domain=ASSOCIATION_DOMAIN, scheme=HEATMAP_SCHEME),
                        title="association"),
        tooltip=["row:N", "column:N", alt.Tooltip("association:Q", format=".2f")],
    ).properties(width=side, height=side)


def heatmaps(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> alt.HConcatChart:
    """Real and synthetic associations side by side, with one shared colour legend."""
    return alt.hconcat(heatmap(real, columns, "Real data"),
                       heatmap(synthetic, columns, "Synthetic data")).resolve_scale(color="shared")


def histogram(frame: pd.DataFrame, column: str) -> alt.Chart:
    """Real and synthetic shares side by side for every bin or level of ``column``."""
    order = frame.drop_duplicates("bin").sort_values("order")["bin"].tolist()
    return alt.Chart(frame, title=column).mark_bar().encode(
        x=alt.X("bin:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
        xOffset=alt.XOffset("data:N", sort=list(SOURCE_COLORS)),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="share of rows"),
        color=alt.Color("data:N", scale=alt.Scale(domain=list(SOURCE_COLORS), range=list(SOURCE_COLORS.values())),
                        title=None),
        tooltip=["bin:N", "data:N", alt.Tooltip("share:Q", format=".1%")],
    ).properties(height=HISTOGRAM_HEIGHT)
