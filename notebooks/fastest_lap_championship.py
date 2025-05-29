"""Build a fastest lap standings visualisation."""

# %%
import datetime
import logging

import fastf1 as ff1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastf1.logger import LoggingManager

# %%
LoggingManager().set_level(logging.ERROR)

# %%
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# %%
results = []
num_races = 0
for race_num in range(1, 25):
    # get the session and check that it's run
    session = ff1.get_session(2025, race_num, "R")
    if session.event.EventDate > datetime.datetime.now():
        print(f"Skipping {session.event.EventName} as it's not happened yet.")
        continue

    # get the session laps
    print(f"Running {session.event.EventName}...")
    session.load(telemetry=False, laps=True, weather=False)
    laps = session.laps

    # take valid laps only
    valid_laps = laps[
        (~laps.LapTime.isna())
        & (~laps.Deleted)
        & (laps.PitOutTime.isna())
        & (laps.PitInTime.isna())
    ].reset_index(drop=True)
    valid_laps.loc[:, "Sector1S"] = valid_laps[
        "Sector1Time"
    ].dt.total_seconds()
    valid_laps.loc[:, "Sector2S"] = valid_laps[
        "Sector2Time"
    ].dt.total_seconds()
    valid_laps.loc[:, "Sector3S"] = valid_laps[
        "Sector3Time"
    ].dt.total_seconds()
    valid_laps.loc[:, "LapTimeS"] = valid_laps["LapTime"].dt.total_seconds()
    valid_laps.loc[:, "IsValidSectors"] = np.isclose(
        valid_laps["Sector1S"]
        + valid_laps["Sector2S"]
        + valid_laps["Sector3S"],
        valid_laps["LapTimeS"],
        rtol=1e-03,
    )
    valid_laps = valid_laps[valid_laps.IsValidSectors].reset_index(drop=True)

    # calculate the fastest laps
    fastest_laps = (
        valid_laps.groupby("Driver", as_index=False)
        .LapTimeS.min()
        .sort_values("LapTimeS", ascending=True)
        .rename(columns={"LapTimeS": "FastestLapTime"})
        .reset_index(drop=True)
    )

    metadata = session.results[
        [
            "Abbreviation",
            "BroadcastName",
            "DriverNumber",
            "TeamName",
            "TeamColor",
            "ClassifiedPosition",
        ]
    ]
    metadata = metadata.rename(columns={"Abbreviation": "Driver"})
    metadata.loc[:, "DSQ"] = metadata.ClassifiedPosition.apply(
        lambda x: True if x == "D" else False
    )
    fastest_laps = fastest_laps.merge(metadata, on="Driver")
    fastest_laps.sort_values(
        ["DSQ", "FastestLapTime"], ascending=[True, True], inplace=True
    )

    fastest_laps["Position"] = np.arange(1, len(fastest_laps) + 1)
    fastest_laps["FastestLapPoints"] = fastest_laps.Position.map(
        POINTS
    ).fillna(0)
    fastest_laps.loc[:, "EventName"] = session.event.EventName

    print(
        f"{session.event.EventName} fastest lap: {fastest_laps.iloc[0].Driver}"
    )
    last_race = session.event.EventName
    last_race_date = session.event.EventDate.strftime("%Y-%m-%d")
    num_races += 1

    results.append(fastest_laps)

# combine all the results
results = pd.concat(results)


# %%
def mode(x):
    """Calculate the modal occurance."""
    return x.mode().iloc[0] if not x.mode().empty else None


standings = (
    results.groupby("Driver", as_index=False)
    .agg(
        {
            "FastestLapPoints": "sum",
            "BroadcastName": mode,
            "DriverNumber": mode,
            "TeamName": mode,
            "TeamColor": mode,
        }
    )
    .sort_values("FastestLapPoints", ascending=False)
    .reset_index(drop=True)
)

# %%
# calculate the number of occurances
countback = (
    results.groupby("Driver", as_index=False)["Position"]
    .value_counts()
    .rename(columns={"count": "PositionAchievements"})
    .pivot(columns="Position", index="Driver", values="PositionAchievements")
    .fillna(0)
)
# zfill to set valid sort order + make Driver col
countback.columns = [
    "Position" + str(col).zfill(2)
    for col in countback.columns.get_level_values(0)
]
countback.reset_index(inplace=True)

# merge countback and sort standing by points and countback occurances
standings = standings.merge(countback, on="Driver")
countback_columns = list(countback.columns)
countback_columns.remove("Driver")
standings.sort_values(
    ["FastestLapPoints"] + sorted(countback_columns),
    inplace=True,
    ascending=False,
)
standings.reset_index(drop=True, inplace=True)

# %%
hp_template = go.layout.Template(
    layout=go.Layout(
        # title
        title=dict(  # noqa: C408
            x=0.07, xref="container", y=0.97, yref="container"
        ),
        # backgrounds and paper
        plot_bgcolor="white",
        paper_bgcolor="white",
        # fonts
        font=dict(family="Arial"),  # noqa: C408
        # axes
        xaxis=dict(  # noqa: C408
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(  # noqa: C408
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        # margin
        margin=dict(l=20, r=20, t=70, b=40),  # noqa: C408
    )
)

# %%
fig = go.Figure()
fig.update_layout(
    template=hp_template,
    width=600,
    height=1000,
)

v_spacing = 5
h_spacing = 2
h_padding = 1
grid_width = 10
grid_thick_vert = 2
grid_thin_vert = 5

# line widths
thick_line = 5
think_line = 3

# adjustments
x_pixel_adjust = 1e-1

# hp styling
hp_red = "#C8102E"
hp_white = "#FFFFFF"
hp_black = "#000000"
xlarge_font = 24
large_font = 20
med_font = 16
small_font = 14

# data locations
position_num = 0.11
name = 0.2
pts = 0.94

# update the title
fig.update_layout(
    title=dict(  # noqa: C408
        text="<b>2025 Drivers Fastest Lap Championship<b>",
        font=dict(size=xlarge_font),  # noqa: C408
    )
)

# update the subtitle
fig.add_annotation(
    text=(
        f"Standings after {num_races} rounds (post {last_race} on "
        f"{last_race_date})."
    ),
    xanchor="left",
    yref="paper",
    align="left",
    x=-2 * x_pixel_adjust,
    y=1.02,
    showarrow=False,
    font=dict(size=small_font),  # noqa: C408
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
    font=dict(color="darkgrey", size=10),  # noqa: C408
)
fig.add_annotation(
    text="High Confidence",
    x=2 * grid_width + h_spacing,
    y=-grid_thin_vert,
    xanchor="right",
    showarrow=False,
    font=dict(color="white", size=10),  # noqa: C408
    bgcolor="#008000",
    bordercolor="black",
    borderpad=2,
)

# get starting location and adjust axes
grid_line_y = len(standings) * v_spacing
fig.update_xaxes(range=[-h_padding, 2 * grid_width + h_spacing + h_padding])
fig.update_yaxes(range=[-v_spacing, grid_line_y + v_spacing])


for i, driver in standings.iterrows():
    # calculate the column
    col = i % 2

    # main grid line
    fig.add_shape(
        type="line",
        x0=col * (grid_width + h_spacing),
        x1=grid_width + (col * (grid_width + h_spacing)),
        y0=grid_line_y,
        y1=grid_line_y,
        line=dict(color="black", width=thick_line),  # noqa: C408
    )

    # vertical LHS of grid box (thick and thin sections)
    fig.add_shape(
        type="line",
        x0=col * (grid_width + h_spacing) + x_pixel_adjust,
        x1=col * (grid_width + h_spacing) + x_pixel_adjust,
        xanchor="right",
        y0=grid_line_y,
        y1=grid_line_y - grid_thick_vert,
        line=dict(color="black", width=thick_line),  # noqa: C408
    )
    fig.add_shape(
        type="line",
        x0=col * (grid_width + h_spacing) + x_pixel_adjust,
        x1=col * (grid_width + h_spacing) + x_pixel_adjust,
        y0=grid_line_y,
        y1=grid_line_y - grid_thin_vert,
        line=dict(color="black", width=think_line),  # noqa: C408
    )

    # vertical RHS of grid box (thick and thin sections)
    fig.add_shape(
        type="line",
        x0=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
        x1=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
        xanchor="right",
        y0=grid_line_y,
        y1=grid_line_y - grid_thick_vert,
        line=dict(color="black", width=thick_line),  # noqa: C408
    )
    fig.add_shape(
        type="line",
        x0=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
        x1=grid_width + (col * (grid_width + h_spacing)) - x_pixel_adjust,
        y0=grid_line_y,
        y1=grid_line_y - grid_thin_vert,
        line=dict(color="black", width=think_line),  # noqa: C408
    )

    # add position number
    pos_num_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
        position_num * grid_width
    )
    text_y = grid_line_y - (grid_thin_vert / 2)
    fig.add_shape(
        type="rect",
        x0=pos_num_x - (6 * position_num),
        x1=pos_num_x + (6 * position_num),
        xanchor="center",
        y0=grid_line_y - (0.15 * grid_thin_vert),
        y1=grid_line_y - (0.85 * grid_thin_vert),
        yanchor="middle",
        line=dict(width=2, color="Black"),  # noqa: C408
        fillcolor=hp_red,
    )
    fig.add_annotation(
        text=i + 1,
        font=dict(size=large_font, color=hp_white),  # noqa: C408
        x=pos_num_x,
        y=text_y,
        xanchor="center",
        yanchor="middle",
        align="center",
        showarrow=False,
    )

    # driver name
    name_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
        name * grid_width
    )
    fig.add_annotation(
        text=driver["BroadcastName"][2:],
        font=dict(size=med_font, color=f"#{driver['TeamColor']}"),  # noqa: C408
        x=name_x,
        y=text_y,
        xanchor="left",
        yanchor="middle",
        align="center",
        showarrow=False,
    )

    # points
    points_x = (col * (grid_width + h_spacing) + x_pixel_adjust) + (
        pts * grid_width
    )
    pts_value = driver["FastestLapPoints"]
    fig.add_annotation(
        text=(
            f"{pts_value:.0f}pts"
            if pts_value.is_integer()
            else f"{pts_value:.1f}pts"
        ),
        font=dict(size=med_font, color=hp_black),  # noqa: C408
        x=points_x,
        y=text_y,
        xanchor="right",
        yanchor="middle",
        align="center",
        showarrow=False,
    )

    # move onto the next grid line
    grid_line_y -= v_spacing

fig.write_image(
    f"../outputs/drivers_fastes_laps_standings_{num_races}_rounds.png"
)
fig.show()

# %%
