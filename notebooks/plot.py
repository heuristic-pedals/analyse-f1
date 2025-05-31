"""analyse-f1 plotting module (currently in dev)."""

from enum import Enum

import pandas as pd
import plotly.graph_objects as go

# hp styling
HP_RED = "#C8102E"
HP_WHITE = "#FFFFFF"
HP_BLACK = "#000000"
XLARGE_FONT = 23
LARGE_FONT = 20
MED_FONT = 16
SMALL_FONT = 14


class GridPlotType(Enum):
    """Valid types for grid plot."""

    DRIVERS = "drivers"
    CONSTRUCTORS = "constructors"

    @classmethod
    def _get_valid(cls):
        """Get the valid grid plot type options."""
        return [member.value for member in cls]


def _hp_grid_plot_template(
    grid_plot_type: str | GridPlotType,
) -> go.layout.Template:
    """Build plotly template for grid plots.

    Parameters
    ----------
    grid_plot_type: str | GridPlotType
        Valid options for grid plot types.

    Returns
    -------
    go.layout.Template
        Plotly template styled for grid plots.
    """
    match grid_plot_type:
        case GridPlotType.DRIVERS.value:
            height = 1000
            title_y = 0.97
        case GridPlotType.CONSTRUCTORS.value:
            height = 575
            title_y = 0.945
        case _:
            raise ValueError(
                f"Invalid `grid_plot_type`: {grid_plot_type}. Must be one of"
                f"{GridPlotType._get_valid()}"
            )

    return go.layout.Template(
        layout=go.Layout(
            # plot sizing
            height=height,
            width=600,
            margin={"l": 20, "r": 20, "t": 70, "b": 40},
            # title configuration
            title={
                "x": 0.069,
                "xref": "container",
                "y": title_y,
                "yref": "container",
                "font": {"size": XLARGE_FONT},
            },
            # background and paper styles
            plot_bgcolor="white",
            paper_bgcolor="white",
            # fonts
            font={"family": "Arial"},
            # axes settings
            xaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
            },
            yaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
            },
        )
    )


def grid_plot(
    standings: pd.DataFrame,
    grid_plot_type: str | GridPlotType,
    points_column: str,
    title: str,
    subtitle: str,
    position_number_offset: float = 0.11,
    name_offset: float = 0.2,
    points_offset: float = 0.94,
    v_spacing: int = 5,
    h_spacing: int = 2,
    h_padding: int = 1,
    grid_width: int = 10,
    grid_thick_vert: int = 2,
    grid_thin_vert: int = 5,
    thick_line: int = 5,
    thin_line: int = 3,
    x_pixel_adjust: float = 1e-1,
) -> go.Figure:
    """Plot a grid standings visualisation.

    Parameters
    ----------
    standings : pd.DataFrame
        Standings to visualise. Must already be sorted for visualisation.
    grid_plot_type : str | GridPlotType
        Type of standings to visualise. Must be one of {"drivers",
        "constructors"}.
    points_column : str
        Column name for standings points.
    title : str
        Visualisation title
    subtitle : str
        Visualisation subtitle
    position_number_offset : float, optional
        Offset for position number, by default 0.11 (11% off LHS of grid line).
    name_offset : float, optional
        Offset for name, by default 0.2 (20% off LHS of grid line).
    points_offset : float, optional
        Points value offset, by default 0.94 (94% off LHS of grid line, right
        aligned).
    v_spacing : int, optional
        Vertical spacing between grid spaces, by default 5.
    h_spacing : int, optional
        Horizontal spacing between grid spaces, by default 2.
    h_padding : int, optional
        Additional horizontal padding to sides of grids, by default 1.
    grid_width : int, optional
        Width of grid spacing, by default 10.
    grid_thick_vert : int, optional
        Size of thick grid downsections, by default 2.
    grid_thin_vert : int, optional
        Size of thin grid downsections, by default 5.
    thick_line : int, optional
        Weight of grid lines, thick, by default 5.
    thin_line : int, optional
        Weight of gird lines, thin, by default 3.
    x_pixel_adjust : float, optional
        Small adjustment to align different pixel sizes, by default 1e-1.

    Returns
    -------
    go.Figure
        Grid standings visualisation

    Raises
    ------
    ValueError
        Invalid type of grid plot.
    KeyError
        Required column is not present in standings.
    """
    # ensure plot type is valid and retrieve specific stylings and template
    match grid_plot_type:
        case GridPlotType.DRIVERS.value:
            subtitle_y_offset = 1.02
            name_col = "BroadcastName"
        case GridPlotType.CONSTRUCTORS.value:
            subtitle_y_offset = 1.035
            name_col = "TeamName"
            # shorten team names for visualisation purposes only
            standings[name_col] = standings[name_col].str.replace(
                r"Racing$|F1 Team$", "", regex=True
            )
        case _:
            raise ValueError(
                f"Invalid `grid_plot_type`: {grid_plot_type}. Must be one of"
                f"{GridPlotType._get_valid()}"
            )
    template = _hp_grid_plot_template(grid_plot_type)

    # check for required columns
    required_columns = [name_col, points_column]
    for col in required_columns:
        if col not in standings.columns:
            raise KeyError(f"Unable to find {col} in standings.")

    # build figure and update title + subtitle
    fig = go.Figure()
    fig.update_layout(
        template=template, title={"text": "<b>" + title + "</b>"}
    )
    fig.add_annotation(
        text=subtitle,
        xanchor="left",
        yref="paper",
        align="left",
        x=-2 * x_pixel_adjust,
        y=subtitle_y_offset,
        showarrow=False,
        font={"size": SMALL_FONT},
    )

    # add sources and confidence
    fig.add_annotation(
        text=(
            "Sources: F1 API/FastF1 v3.5.3 (collection), "
            "Grid Graphs (analysis and visualisation)."
        ),
        x=-x_pixel_adjust,
        y=-grid_thin_vert,
        xanchor="left",
        align="center",
        showarrow=False,
        font={"color": "darkgrey", "size": 10},
    )
    fig.add_annotation(
        text="High Confidence",
        x=2 * grid_width + h_spacing,
        y=-grid_thin_vert,
        xanchor="right",
        showarrow=False,
        font={"color": HP_WHITE, "size": 10},
        bgcolor="#008000",
        bordercolor="black",
        borderpad=2,
    )

    # get starting location and adjust axes to fit view
    grid_line_y = len(standings) * v_spacing
    fig.update_xaxes(
        range=[-h_padding, 2 * grid_width + h_spacing + h_padding]
    )
    fig.update_yaxes(range=[-v_spacing, grid_line_y + v_spacing])

    # add each place within the standings
    for i, place in standings.iterrows():
        # calculate the column
        col = i % 2

        # main grid line
        fig.add_shape(
            type="line",
            x0=col * (grid_width + h_spacing),
            x1=grid_width + (col * (grid_width + h_spacing)),
            y0=grid_line_y,
            y1=grid_line_y,
            line={"color": "black", "width": thick_line},
        )

        # vertical LHS of grid box (thick and thin sections)
        fig.add_shape(
            type="line",
            x0=col * (grid_width + h_spacing) + x_pixel_adjust,
            x1=col * (grid_width + h_spacing) + x_pixel_adjust,
            xanchor="right",
            y0=grid_line_y,
            y1=grid_line_y - grid_thick_vert,
            line={"color": "black", "width": thick_line},
        )
        fig.add_shape(
            type="line",
            x0=col * (grid_width + h_spacing) + x_pixel_adjust,
            x1=col * (grid_width + h_spacing) + x_pixel_adjust,
            y0=grid_line_y,
            y1=grid_line_y - grid_thin_vert,
            line={"color": "black", "width": thin_line},
        )

        # vertical RHS of grid box (thick and thin sections)
        fig.add_shape(
            type="line",
            x0=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
            x1=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
            xanchor="right",
            y0=grid_line_y,
            y1=grid_line_y - grid_thick_vert,
            line={"color": "black", "width": thick_line},
        )
        fig.add_shape(
            type="line",
            x0=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
            x1=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
            y0=grid_line_y,
            y1=grid_line_y - grid_thin_vert,
            line={"color": "black", "width": thin_line},
        )

        # add position number
        pos_num_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
            position_number_offset * grid_width
        )
        text_y = grid_line_y - (grid_thin_vert / 2)
        fig.add_shape(
            type="rect",
            x0=pos_num_x - (6 * position_number_offset),
            x1=pos_num_x + (6 * position_number_offset),
            xanchor="center",
            y0=grid_line_y - (0.15 * grid_thin_vert),
            y1=grid_line_y - (0.85 * grid_thin_vert),
            yanchor="middle",
            line={"width": 2, "color": "Black"},
            fillcolor=HP_RED,
        )
        fig.add_annotation(
            text=i + 1,
            font={"size": LARGE_FONT, "color": HP_WHITE},
            x=pos_num_x,
            y=text_y,
            xanchor="center",
            yanchor="middle",
            align="center",
            showarrow=False,
        )

        # add driver/team name
        name_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
            name_offset * grid_width
        )
        place_name = (
            place[name_col][2:]
            if grid_plot_type == GridPlotType.DRIVERS.value
            else place[name_col]
        )
        # correct team colour if it is missing
        team_color = f"#{place['TeamColor']}"
        if team_color.lower() == "#ffffff":
            if "Haas" in place["TeamName"]:
                team_color = "#A9A9A9"  # 2021 Haas
            elif "AlphaTauri" in place["TeamName"]:
                team_color = "#C8C8C8"  # 2020 Alpha Tauri
            elif "Williams" in place["TeamName"]:
                team_color = "#00A0DE"  # 2019 Williams
            else:
                raise ValueError(
                    f"{place['TeamName']} got {team_color} but have no "
                    "information to correct for plotting on white backaground."
                )
        fig.add_annotation(
            text=place_name,
            font={"size": MED_FONT, "color": team_color},
            x=name_x,
            y=text_y,
            xanchor="left",
            yanchor="middle",
            align="center",
            showarrow=False,
        )

        # add points value
        points_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
            points_offset * grid_width
        )
        pts_value = place[points_column]
        fig.add_annotation(
            text=(
                f"{pts_value:.0f}pts"
                if pts_value.is_integer()
                else f"{pts_value:.1f}pts"
            ),
            font={"size": MED_FONT, "color": HP_BLACK},
            x=points_x,
            y=text_y,
            xanchor="right",
            yanchor="middle",
            align="center",
            showarrow=False,
        )

        # move onto the next grid line
        grid_line_y -= v_spacing

    return fig
