"""Comparising Miami 2025 SQ3 times.

All work in this file is a work in progress. All methods and plots will be
added to `analyse_f1` overtime. It is used for development purposes
only, so comments and docstrings are initially minimal.
"""

# %%
import datetime

import fastf1 as ff1
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import integrate

# %%
session = ff1.get_session(2025, "miami", "SQ")
session.load(telemetry=True, laps=True, weather=False)

# %%
TRACK_LENGTH = 5410

# %%
comparitor_driver = "ANT"
other_drivers = ["PIA", "NOR", "VER"]

# check comparitor and other drivers are valid
valid_abbreviations = session.results.Abbreviation.unique()
if comparitor_driver not in valid_abbreviations:
    raise ValueError(
        f"{comparitor_driver} is not a valid driver abbreivation."
    )
for driver in other_drivers:
    if driver not in valid_abbreviations:
        raise ValueError(f"{driver} is not a valid driver abbreviation.")


# %%
def get_driver_fastest_lap(
    session,
    driver_abbrev,
) -> pd.DataFrame:
    """Get fastest lap for driver, calculating distance through lap."""
    during_lap = (
        session.laps.pick_drivers(driver_abbrev).pick_fastest().get_car_data()
    )[["Time", "Speed"]]
    during_lap["Time"] = during_lap["Time"].dt.total_seconds()
    start_lap = pd.DataFrame(
        {
            "Time": 0,
            "Speed": during_lap.Speed.iloc[0],
        },
        index=[0],
    )
    fl_metadata = session.laps.pick_drivers(driver_abbrev).pick_fastest()
    end_lap = pd.DataFrame(
        {
            "Time": fl_metadata.LapTime.total_seconds(),
            "Speed": fl_metadata.SpeedFL,
        },
        index=[0],
    )
    fastest_lap = pd.concat([start_lap, during_lap, end_lap]).reset_index(
        drop=True
    )
    fastest_lap["Distance"] = integrate.cumulative_simpson(
        y=fastest_lap.Speed * 0.277778, x=fastest_lap.Time, initial=0
    )
    fastest_lap["Distance"] = (
        fastest_lap["Distance"] / fastest_lap["Distance"].max()
    ) * TRACK_LENGTH
    fastest_lap.set_index("Distance", inplace=True)

    return fastest_lap


comparitor_fl_df = get_driver_fastest_lap(session, comparitor_driver)
other_fl_dfs = {
    driver_abbrev: get_driver_fastest_lap(session, driver_abbrev)
    for driver_abbrev in other_drivers
}

# %%
unique_distances = (
    pd.concat(
        [
            comparitor_fl_df.index.to_series(),
            *[x.index.to_series() for x in other_fl_dfs.values()],
        ]
    )
    .sort_values()
    .unique()
)


# %%
def reindex_and_interpolate(
    df: pd.DataFrame, new_index: pd.Series, **interpolate_kwargs: dict
) -> pd.DataFrame:
    """Reindex to common index and interpolate telemetry data."""
    return df.reindex(new_index).interpolate(**interpolate_kwargs)


interpolation_args = {
    "method": "piecewise_polynomial",
    "limit_direction": "both",
}

comparitor_interpolated_fl_df = reindex_and_interpolate(
    comparitor_fl_df, unique_distances, **interpolation_args
)
other_interpolated_fl_dfs = {
    driver_abbrev: reindex_and_interpolate(
        other_fl_df, unique_distances, **interpolation_args
    )
    for driver_abbrev, other_fl_df in other_fl_dfs.items()
}

# %%
for driver, df in other_interpolated_fl_dfs.items():
    df[f"Time difference to {comparitor_driver} (s)"] = (
        df["Time"] - comparitor_interpolated_fl_df["Time"]
    )
    df[f"Speed difference to {comparitor_driver} (Km/h)"] = (
        comparitor_interpolated_fl_df["Speed"] - df["Speed"]
    )
    df.loc[:, "Driver"] = driver

# %%
other_interpolated_fl_df = pd.concat(list(other_interpolated_fl_dfs.values()))

# %%
# Define the custom template with a name
hp_line_template = go.layout.Template(
    layout=go.Layout(
        title=dict(  # noqa: C408
            x=0.065,
            xref="container",
        ),
        # White background
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Font settings
        font=dict(family="Arial"),  # noqa: C408
        # Grid lines
        xaxis=dict(  # noqa: C408
            showgrid=False,
            tickformat=",.0f",
            title=dict(standoff=8),  # noqa: C408
        ),
        yaxis=dict(  # noqa: C408
            showgrid=True,
            gridcolor="lightgray",
            linecolor="white",
        ),
        margin=dict(l=40, r=30, t=85, b=85),  # noqa: C408
    )
)

# %%
corners = session.get_circuit_info().corners[["Number", "Distance"]]

# %%
fig = px.line(
    other_interpolated_fl_df,
    y=f"Speed difference to {comparitor_driver} (Km/h)",
    color="Driver",
    color_discrete_sequence=["#FF8000", "#0093DD", "#00174C"],
    labels={
        "Distance": "Distance around lap (m)",
        f"Speed difference to {comparitor_driver} (Km/h)": (
            "Speed Difference (Km/h)"
        ),
    },
    title=(
        f"<b>Car speed differences to {comparitor_driver} during Miami SQ3 "
        "fastest laps</b>"
    ),
    template=hp_line_template,
)

fig.update_yaxes(range=[-31, 39], linecolor="white")
fig.update_layout(
    yaxis_title=None,
    annotations=[
        dict(  # noqa: C408
            text="Speed difference (Km/h)",
            xref="paper",
            yref="paper",
            x=0,
            y=1,  # Position the text
            showarrow=False,
            font=dict(size=14),  # noqa: C408
        ),
        dict(  # noqa: C408
            text=(
                "ANT had consistent straight line speed over VER, a NOR "
                "mistake around T6/T7 gave ANT a large gap, and PIA seemed to "
                "<br>brake too late into the last corner. Altogether, this "
                "gave ANT pole position for the sprint race."
            ),
            xref="paper",
            yref="paper",
            align="left",
            x=0,
            y=1.14,  # Position the text
            showarrow=False,
            font=dict(size=10),  # noqa: C408
        ),
    ],
    legend=dict(  # noqa: C408
        orientation="h",  # Horizontal orientation
        yanchor="bottom",
        y=0.94,  # Adjust the y position to be at the top
        xanchor="right",
        x=1,  # Adjust the x position to be at the right
        valign="middle",
    ),
)

corner_y_max = 30
corner_y_min = 25
higher_corners = []
skip_corners = [12, 14, 15]
for _, corner in corners.iterrows():
    if corner.Number in higher_corners:
        y1 = corner_y_max + 2.5
    else:
        y1 = corner_y_max
    fig.add_shape(
        type="line",
        x0=corner.Distance,
        x1=corner.Distance,
        y0=corner_y_min,
        y1=y1,
        line_width=2,
        line_dash="dot",
        line_color="darkgrey",
    )
    if corner.Number in skip_corners:
        continue
    fig.add_annotation(
        text=int(corner.Number),
        x=corner.Distance,
        xanchor="center",
        y=y1 + 2,
        showarrow=False,
    )

fig.add_trace(
    go.Scatter(
        x=[2000, 2000],
        y=[-10.5, -19.5],
        marker=dict(  # noqa: C408
            size=9,
            symbol="arrow-bar-up",
            angleref="previous",
            color="darkgrey",
        ),
        showlegend=False,
    ),
)
fig.add_annotation(
    text=f"{comparitor_driver} slower",
    x=2040,
    y=-15,
    showarrow=False,
    xanchor="left",
)
fig.add_trace(
    go.Scatter(
        x=[2000, 2000],
        y=[10.5, 19.5],
        marker=dict(  # noqa: C408
            size=9,
            symbol="arrow-bar-up",
            angleref="previous",
            color="darkgrey",
        ),
        showlegend=False,
    ),
)
fig.add_annotation(
    text=f"{comparitor_driver} faster",
    x=2040,
    y=15,
    showarrow=False,
    xanchor="left",
)
fig.add_annotation(
    text="Moderate Confidence",
    x=1,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#FF5F15",
    bordercolor="black",
    borderpad=2,
)
fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (analysis and visualisation)."
    ),
    x=0,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

output_height = 460
output_width = 640
fname_prefix = f"{comparitor_driver.lower()}_speed_differences"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)
fig.show()

# %%
fig = px.line(
    other_interpolated_fl_df,
    y=f"Time difference to {comparitor_driver} (s)",
    color="Driver",
    color_discrete_sequence=["#FF8000", "#0093DD", "#00174C"],
    labels={
        "Distance": "Distance around lap (m)",
        f"Time difference to {comparitor_driver} (s)": ("Time difference (s)"),
    },
    title=(
        f"<b>Car time differences to {comparitor_driver} during Miami SQ3 "
        "fastest laps</b>"
    ),
    template=hp_line_template,
)

fig.update_yaxes(range=[-0.21, 0.49], linecolor="white")
fig.update_layout(
    yaxis_title=None,
    annotations=[
        dict(  # noqa: C408
            text="Time difference (s)",
            xref="paper",
            yref="paper",
            x=0,
            y=1,  # Position the text
            showarrow=False,
            font=dict(size=14),  # noqa: C408
        ),
        dict(  # noqa: C408
            text=(
                "VER's speed advantage through the corner sections wasn't "
                "enough to counteract straight line time losses to ANT. "
                "NOR never<br>recovered from a T6/T7 mistake, and PIA's "
                "T17 exit proved to be the pivotal momement versus ANT."
            ),
            xref="paper",
            yref="paper",
            align="left",
            x=0,
            y=1.14,  # Position the text
            showarrow=False,
            font=dict(size=10),  # noqa: C408
        ),
    ],
    legend=dict(  # noqa: C408
        orientation="h",  # Horizontal orientation
        yanchor="bottom",
        y=0.94,  # Adjust the y position to be at the top
        xanchor="right",
        x=1,  # Adjust the x position to be at the right
        valign="middle",
    ),
)

corner_y_max = 0.4
corner_y_min = 0.35
higher_corners = []
skip_corners = [12, 14, 15]
for _, corner in corners.iterrows():
    if corner.Number in higher_corners:
        y1 = corner_y_max + 0.1
    else:
        y1 = corner_y_max
    fig.add_shape(
        type="line",
        x0=corner.Distance,
        x1=corner.Distance,
        y0=corner_y_min,
        y1=y1,
        line_width=2,
        line_dash="dot",
        line_color="darkgrey",
    )
    if corner.Number in skip_corners:
        continue
    fig.add_annotation(
        text=int(corner.Number),
        x=corner.Distance,
        xanchor="center",
        y=y1 + 0.02,
        showarrow=False,
    )

fig.add_trace(
    go.Scatter(
        x=[3000, 3000],
        y=[-0.11, -0.19],
        marker=dict(  # noqa: C408
            size=9,
            symbol="arrow-bar-up",
            angleref="previous",
            color="darkgrey",
        ),
        showlegend=False,
    ),
)
fig.add_annotation(
    text=f"{comparitor_driver} slower",
    x=3040,
    y=-0.15,
    showarrow=False,
    xanchor="left",
)
fig.add_trace(
    go.Scatter(
        x=[2000, 2000],
        y=[0.21, 0.29],
        marker=dict(  # noqa: C408
            size=9,
            symbol="arrow-bar-up",
            angleref="previous",
            color="darkgrey",
        ),
        showlegend=False,
    ),
)
fig.add_annotation(
    text=f"{comparitor_driver} faster",
    x=2040,
    y=0.25,
    showarrow=False,
    xanchor="left",
)
fig.add_annotation(
    text="Moderate Confidence",
    x=1,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#FF5F15",
    bordercolor="black",
    borderpad=2,
)
fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (analysis and visualisation)."
    ),
    x=0,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

driver_data = "NOR"
data = other_interpolated_fl_dfs[driver_data].iloc[-1]
last_distance = other_interpolated_fl_dfs[driver_data].index[-1]
last_time_delta = data[f"Time difference to {comparitor_driver} (s)"]
fig.add_annotation(
    text=f"+{last_time_delta:.3f}",
    x=0.97,
    y=last_time_delta + 0.02,
    xref="paper",
    xanchor="left",
    align="left",
    showarrow=False,
    font=dict(color="#0093DD", size=10),  # noqa: C408
)

driver_data = "PIA"
data = other_interpolated_fl_dfs[driver_data].iloc[-1]
last_distance = other_interpolated_fl_dfs[driver_data].index[-1]
last_time_delta = data[f"Time difference to {comparitor_driver} (s)"]
fig.add_annotation(
    text=f"+{last_time_delta:.3f}",
    x=0.97,
    y=last_time_delta + 0.02,
    xref="paper",
    xanchor="left",
    align="left",
    showarrow=False,
    font=dict(color="#FF8000", size=10),  # noqa: C408
)

driver_data = "VER"
data = other_interpolated_fl_dfs[driver_data].iloc[-1]
last_distance = other_interpolated_fl_dfs[driver_data].index[-1]
last_time_delta = data[f"Time difference to {comparitor_driver} (s)"]
fig.add_annotation(
    text=f"+{last_time_delta:.3f}",
    x=0.97,
    y=last_time_delta + 0.02,
    xref="paper",
    xanchor="left",
    align="left",
    showarrow=False,
    font=dict(color="#00174C", size=10),  # noqa: C408
)

output_height = 460
output_width = 640
fname_prefix = f"{comparitor_driver.lower()}_time_differences"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)
fig.show()

# %%
lap = session.laps.pick_fastest()
pos = lap.get_pos_data()

circuit_info = session.get_circuit_info()


def rotate(xy, *, angle):
    """Rotate x, y coords to align with relative angle."""
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


# Get an array of shape [n, 2] where n is the number of points and the second
# axis is x and y.
track = pos.loc[:, ("X", "Y")].to_numpy()

# Convert the rotation angle from degrees to radian.
track_angle = circuit_info.rotation / 180 * np.pi

# Rotate and plot the track map.
rotated_track = rotate(track, angle=track_angle)
rotated_track = pd.DataFrame(rotated_track, columns=["X", "Y"])

# %%
fig = px.line(
    rotated_track,
    x="X",
    y="Y",
    template=hp_line_template,
    color_discrete_sequence=["#000000"],
    title="<b>Miami International Autodrome, Florida, USA</b>",
)

fig.update_traces(line={"width": 5})
fig.update_yaxes(visible=False, showgrid=False)
fig.update_xaxes(visible=False, showgrid=False)

offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

# Iterate over all corners.
for _, corner in circuit_info.corners.iterrows():
    # Create a string from corner number and letter
    txt = f"{corner['Number']}{corner['Letter']}"

    # Convert the angle from degrees to radian.
    offset_angle = corner["Angle"] / 180 * np.pi

    # Rotate the offset vector so that it points sideways from the track.
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

    # Add the offset to the position of the corner
    text_x = corner["X"] + offset_x
    text_y = corner["Y"] + offset_y

    # Rotate the text position equivalently to the rest of the track map
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)

    # Rotate the center of the corner equivalently to the rest of the track map
    track_x, track_y = rotate([corner["X"], corner["Y"]], angle=track_angle)

    radius_x = 325
    radius_y = 275
    fig.add_shape(
        type="circle",
        x0=text_x - radius_x,
        y0=text_y - radius_y,
        xanchor="left",
        yanchor="left",
        x1=text_x + radius_x,
        y1=text_y + radius_y,
        line=dict(width=2, color="Black"),  # noqa: C408
        fillcolor="#C8102E",
    )
    fig.add_annotation(
        text=txt,
        x=text_x,
        y=text_y,
        showarrow=False,
        align="center",
        font=dict(color="white", size=14),  # noqa: C408
    )

fig.add_annotation(
    text=(
        "Miami GP 2025 circuit map, constructed using ANT's car position "
        "data during their SQ3 pole lap."
    ),
    xref="paper",
    yref="paper",
    align="left",
    x=0,
    y=1.14,  # Position the text
    showarrow=False,
    font=dict(size=10),  # noqa: C408
)

fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (visualisation)."
    ),
    x=0,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

fig.add_annotation(
    text="High Confidence",
    x=1,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#008000",
    bordercolor="black",
    borderpad=2,
)

output_height = 460
output_width = 640
fname_prefix = "miami_track_map"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)

fig.show()

# %%
laps = session.laps

sector_times = []
for sector in range(1, 4):
    sector_time = pd.DataFrame(
        laps.iloc[laps[f"Sector{sector}Time"].idxmin()]
    ).T[["Driver", f"Sector{sector}Time"]]
    sector_time.rename(
        columns={f"Sector{sector}Time": "Time (s)"}, inplace=True
    )
    sector_time["Time (s)"] = sector_time["Time (s)"].apply(
        lambda x: x.total_seconds()
    )
    sector_time.loc[:, "Sector"] = sector
    sector_times.append(sector_time[["Sector", "Driver", "Time (s)"]])

sector_times = pd.concat(sector_times).reset_index(drop=True)

sector_times.loc[:, "Sector Description"] = [
    "Start line to T8 exit",
    "T8 exit to T16 exit",
    "T16 exit to finish line",
]


# %%
def mark_sectors(idx: int) -> str:
    """Highlight sectors."""
    if idx < 103:
        return 1
    elif idx < 232:
        return 2
    else:
        return 3


rotated_track["sector"] = rotated_track.index.map(mark_sectors)
sect_1_extension = rotated_track.iloc[[103]]
sect_1_extension.index = [102]
sect_1_extension.loc[:, "sector"] = 1

sect_2_extension = rotated_track.iloc[[232]]
sect_2_extension.index = [231]
sect_2_extension.loc[:, "sector"] = 2

sect_3_extension = rotated_track.iloc[[0]]
sect_3_extension.index = [rotated_track.index.max() + 1]
sect_3_extension.loc[:, "sector"] = 3

plot_rotated_track = (
    pd.concat(
        [rotated_track, sect_1_extension, sect_2_extension, sect_3_extension]
    )
    .copy()
    .sort_index()
)

fig = px.line(
    plot_rotated_track,
    x="X",
    y="Y",
    color="sector",
    template=hp_line_template,
    color_discrete_sequence=["#00174C", "#FF8000", "#0093DD"],
    title="<b>Fastest sector times during SQ3</b>",
)

fig.update_layout(showlegend=False)
fig.update_traces(line={"width": 5})
fig.update_yaxes(visible=False, showgrid=False)
fig.update_xaxes(visible=False, showgrid=False)

offset_vector = [500, 0]  # offset length is chosen arbitrarily to 'look good'

# Iterate over all corners.
for _, corner in circuit_info.corners.iterrows():
    # Create a string from corner number and letter
    txt = f"{corner['Number']}{corner['Letter']}"

    # Convert the angle from degrees to radian.
    offset_angle = corner["Angle"] / 180 * np.pi

    # Rotate the offset vector so that it points sideways from the track.
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

    # Add the offset to the position of the corner
    text_x = corner["X"] + offset_x
    text_y = corner["Y"] + offset_y

    # Rotate the text position equivalently to the rest of the track map
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)

    # Rotate the center of the corner equivalently to the rest of the track map
    track_x, track_y = rotate([corner["X"], corner["Y"]], angle=track_angle)

    radius_x = 325
    radius_y = 275
    fig.add_shape(
        type="circle",
        x0=text_x - radius_x,
        y0=text_y - radius_y,
        xanchor="left",
        yanchor="left",
        x1=text_x + radius_x,
        y1=text_y + radius_y,
        line=dict(width=2, color="Black"),  # noqa: C408
        fillcolor="#C8102E",
    )
    fig.add_annotation(
        text=txt,
        x=text_x,
        y=text_y,
        showarrow=False,
        align="center",
        font=dict(color="white", size=14),  # noqa: C408
    )

fig.add_annotation(
    text=(
        "VER, PIA, and, NOR each took a fastest sector time during SQ3 "
        "but failed to take pole position."
    ),
    xref="paper",
    yref="paper",
    align="left",
    x=0,
    y=1.1,  # Position the text
    showarrow=False,
    font=dict(size=10),  # noqa: C408
)

fig.add_annotation(
    text=(
        f"<b>Sector 1</b><br>{sector_times.Driver.iloc[0]}: "
        f"{sector_times['Time (s)'].iloc[0]:.3f}s"
    ),
    x=1800,
    y=-2000,
    align="center",
    showarrow=False,
    font=dict(color="#00174C", size=14),  # noqa: C408
)

fig.add_annotation(
    text=(
        f"<b>Sector 2</b><br>{sector_times.Driver.iloc[1]}: "
        f"{sector_times['Time (s)'].iloc[1]:.3f}s"
    ),
    x=8500,
    y=-4400,
    align="center",
    showarrow=False,
    font=dict(color="#FF8000", size=14),  # noqa: C408
)

fig.add_annotation(
    text=(
        f"<b>Sector 3</b><br>{sector_times.Driver.iloc[2]}: "
        f"{sector_times['Time (s)'].iloc[2]:.3f}s"
    ),
    x=3300,
    y=1700,
    align="center",
    showarrow=False,
    font=dict(color="#0093DD", size=14),  # noqa: C408
)

fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (analysis and visualisation)."
    ),
    x=0,
    y=-0.1,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

fig.add_annotation(
    text="High Confidence",
    x=1,
    y=-0.1,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#008000",
    bordercolor="black",
    borderpad=2,
)

fig.update_layout(margin=dict(l=40, r=40, t=70, b=60))  # noqa: C408

output_height = 460
output_width = 640
fname_prefix = "miami_sector_map"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)

fig.show()

# %%
fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[0.2, 0.2, 0.2, 0.4],
            header=dict(  # noqa: C408
                values=["Sector", "Driver", "Time (s)", "Sector Description"],
                fill_color="#C8102E",
                font=dict(color="white", size=14, weight="bold"),  # noqa: C408
                height=25,
            ),
            cells=dict(  # noqa: C408
                values=[
                    sector_times.Sector,
                    sector_times.Driver,
                    sector_times["Time (s)"].round(3),
                    sector_times["Sector Description"],
                ],
                fill=dict(  # noqa: C408
                    color=[
                        ["#FFFFFF"] * 3,
                        ["#00174C", "#FF8000", "#0093DD"],
                        ["#FFFFFF"] * 3,
                        ["#FFFFFF"] * 3,
                    ],
                ),
                font=dict(  # noqa: C408
                    size=14,
                    color=[
                        ["#000000"] * 3,
                        ["#FFC906", "#000000", "#FFFFFF"],
                        ["#000000"] * 3,
                        ["#000000"] * 3,
                    ],
                ),
                height=25,
                align=["center", "center"],
            ),
        )
    ],
)

output_height = 255
output_width = 640
fig.update_layout(
    title=dict(text="<b>Fastest sector times during SQ3</b>"),  # noqa: C408
    template=hp_line_template,
    height=output_height,
    width=output_width,
    margin=dict(l=40, r=30, t=75, b=60),  # noqa: C408
)

fig.add_annotation(
    text=(
        "VER, PIA, and, NOR each took a fastest sector time during SQ3 "
        "but failed to take pole position."
    ),
    xref="paper",
    yref="paper",
    align="left",
    x=0,
    y=1.28,  # Position the text
    showarrow=False,
    font=dict(size=10),  # noqa: C408
)

fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (analysis and visualisation)."
    ),
    x=0,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

fig.add_annotation(
    text="High Confidence",
    x=1,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#008000",
    bordercolor="black",
    borderpad=2,
)

fname_prefix = "fastest_sectors_table"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)

fig.show()
# %%

occurances = []
total_sessions = 0
unloadable_sessions = []
for year in range(2018, 2026):
    print(f"Working on year {year}...")
    events = ff1.get_event_schedule(year)
    events = events[events.EventFormat != "testing"]
    for round_num in events.RoundNumber:
        if events[events.RoundNumber == round_num].EventDate.values[
            0
        ] >= pd.Timestamp(datetime.datetime.now()):
            print(f"Skipping {year}, {round_num} as it's in the future")
            continue
        for quali_type in ["Q", "SQ"]:
            try:
                session = ff1.get_session(year, round_num, quali_type)
            except ValueError as e:
                if "Session type 'SQ' does not exist for this event" in str(e):
                    print(f"Skipping SQ for {year}, {round_num}...")
                    continue
                if "Cannot get testing event by round number!" in str(e):
                    print(f"Skipping test event {year}, {round_num}...")
                else:
                    raise e
            total_sessions += 1
            session.load(telemetry=False, laps=True, weather=False)
            try:
                laps = session.laps[session.laps.IsAccurate].reset_index(
                    drop=True
                )
            except ff1.core.DataNotLoadedError:
                print(
                    f"Skipping {year}, {round_num}, {quali_type} as data is "
                    "missing"
                )
                unloadable_sessions.append([year, round_num, quali_type])

            pole_driver = laps.loc[laps.LapTime.idxmin()].Driver

            sector_times = []
            for sector in range(1, 4):
                sector_time = pd.DataFrame(
                    laps.iloc[laps[f"Sector{sector}Time"].idxmin()]
                ).T[["Driver", f"Sector{sector}Time"]]
                sector_time.rename(
                    columns={f"Sector{sector}Time": "Time (s)"}, inplace=True
                )
                sector_time["Time (s)"] = sector_time["Time (s)"].apply(
                    lambda x: x.total_seconds()
                )
                sector_time.loc[:, "Sector"] = sector
                sector_times.append(
                    sector_time[["Sector", "Driver", "Time (s)"]]
                )
            sector_times = pd.concat(sector_times).reset_index(drop=True)

            if pole_driver not in sector_times.Driver.unique():
                print(f"Adding: {year}, {round_num}, {pole_driver}")
                occurances.append(
                    [
                        year,
                        events[
                            events.RoundNumber == round_num
                        ].EventName.values[0],
                        pole_driver,
                    ]
                )

# %%
occurances_df = pd.DataFrame(
    occurances, columns=["Year", "EventName", "Driver"]
)

occurances_df = (
    occurances_df.groupby("Driver", as_index=False)["Year"]
    .count()
    .rename(columns={"Year": "Number of Occurances"})
    .sort_values("Number of Occurances")
)

# %%
fig = px.bar(
    occurances_df,
    x="Number of Occurances",
    y="Driver",
    template=hp_line_template,
    color_discrete_sequence=["#C8102E"],
    title="<b>Pole position drivers without setting a fastest sector time</b>",
)

fig.update_yaxes(
    showgrid=False,
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="lightgray",
)
fig.update_layout(yaxis_title=None, xaxis=dict(domain=[0.05, 1]))  # noqa: C408

fig.add_annotation(
    text=(
        "Since 2018, 8 different drivers have obtained"
        " pole position without setting a fastest sector time during their"
        " pole lap (across<br>169 qualifying sessions, including sprints). VER "
        "has done this 7 times, the most of all drivers during this"
        " time period."
    ),
    xref="paper",
    yref="paper",
    align="left",
    x=0,
    y=1.14,  # Position the text
    showarrow=False,
    font=dict(size=10),  # noqa: C408
)

fig.add_annotation(
    text=(
        "Sources: F1 API/FastF1 v3.5.3 (collection), "
        "Grid Graphs (visualisation)."
    ),
    x=0,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="left",
    align="center",
    showarrow=False,
    font=dict(color="darkgrey", size=10),  # noqa: C408
)

fig.add_annotation(
    text="High Confidence",
    x=1,
    y=-0.21,
    xref="paper",
    yref="paper",
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#008000",
    bordercolor="black",
    borderpad=2,
)

output_height = 460
output_width = 640
fname_prefix = "no_fastest_sectors"
fig.write_image(
    f"../outputs/{fname_prefix}.svg",
    height=output_height,
    width=output_width,
)
fig.write_image(
    f"../outputs/{fname_prefix}.png",
    height=output_height,
    width=output_width,
)

fig.show()

# %%
