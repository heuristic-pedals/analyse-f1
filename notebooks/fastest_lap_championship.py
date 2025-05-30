"""Build a fastest lap standings visualisation."""

# %%
import datetime
import logging

import fastf1 as ff1
import numpy as np
import pandas as pd
from fastf1.logger import LoggingManager
from plot import grid_plot

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


drivers_standings = (
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

constructors_standings = (
    results.groupby("TeamName", as_index=False)
    .agg({"FastestLapPoints": "sum", "TeamColor": mode})
    .sort_values("FastestLapPoints", ascending=False)
    .reset_index(drop=True)
)

# %%
# calculate the number of occurances
driver_countback = (
    results.groupby("Driver", as_index=False)["Position"]
    .value_counts()
    .rename(columns={"count": "PositionAchievements"})
    .pivot(columns="Position", index="Driver", values="PositionAchievements")
    .fillna(0)
)
# zfill to set valid sort order + make Driver col
driver_countback.columns = [
    "Position" + str(col).zfill(2)
    for col in driver_countback.columns.get_level_values(0)
]
driver_countback.reset_index(inplace=True)

# merge countback and sort standing by points and countback occurances
drivers_standings = drivers_standings.merge(driver_countback, on="Driver")
driver_countback_columns = list(driver_countback.columns)
driver_countback_columns.remove("Driver")
drivers_standings.sort_values(
    ["FastestLapPoints"] + sorted(driver_countback_columns),
    inplace=True,
    ascending=False,
)
drivers_standings.reset_index(drop=True, inplace=True)

# %%
constructors_countback = (
    results.groupby("TeamName", as_index=False)["Position"]
    .value_counts()
    .rename(columns={"count": "PositionAchievements"})
    .pivot(columns="Position", index="TeamName", values="PositionAchievements")
    .fillna(0)
)
# zfill to set valid sort order + make Driver col
constructors_countback.columns = [
    "Position" + str(col).zfill(2)
    for col in constructors_countback.columns.get_level_values(0)
]
constructors_countback.reset_index(inplace=True)

# merge countback and sort standing by points and countback occurances
constructors_standings = constructors_standings.merge(
    constructors_countback, on="TeamName"
)
constructors_countback_columns = list(constructors_countback.columns)
constructors_countback_columns.remove("TeamName")
constructors_standings.sort_values(
    ["FastestLapPoints"] + sorted(constructors_countback_columns),
    inplace=True,
    ascending=False,
)
constructors_standings.reset_index(drop=True, inplace=True)

# %%
# plot drivers standings
fig_drivers = grid_plot(
    drivers_standings,
    "drivers",
    "FastestLapPoints",
    "2025 Drivers Fastest Lap Championship",
    f"Standings after {num_races} rounds (post {last_race} on "
    f"{last_race_date}).",
)
fig_drivers.write_image(
    f"../outputs/drivers_fastest_laps_standings_{num_races}_rounds.png"
)
fig_drivers.show()

# %%
# plot constructors standings
fig_constructors = grid_plot(
    constructors_standings,
    "constructors",
    "FastestLapPoints",
    "2025 Constructors Fastest Lap Championship",
    f"Standings after {num_races} rounds (post {last_race} on "
    f"{last_race_date}).",
)
fig_constructors.write_image(
    f"../outputs/constructors_fastest_laps_standings_{num_races}_rounds.png"
)
fig_constructors.show()
