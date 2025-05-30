"""Build a fastest lap standings visualisation."""

import argparse
import datetime
import logging
import sys

import fastf1 as ff1
import numpy as np
import pandas as pd
from fastf1.logger import LoggingManager
from plot import grid_plot


class CLI:
    """Handle the cli inputs."""

    def __init__(self):
        """Parse the command line arguments."""
        parser = argparse.ArgumentParser(
            description=(
                "Build a F1 fastest lap championship standings for drivers and"
                " constructors across a season."
            )
        )

        # set the required arguments
        required_args = [
            {
                "name": "year",
                "help": (
                    "The F1 season year. Must be >= 2018 and not in the "
                    "future."
                ),
                "type": self.year,
            }
        ]
        for arg in required_args:
            parser.add_argument(
                arg["name"], help=arg["help"], type=arg["type"]
            )

        # handle the optional args
        optional_ags = [
            {
                "name": "--log-level",
                "help": (
                    "Logging level. Must be one of 'debug', 'info', 'warning',"
                    " 'error', or 'critical'."
                ),
                "default": "info",
                "type": self.log_level,
            }
        ]
        for arg in optional_ags:
            parser.add_argument(
                arg["name"],
                help=arg["help"],
                type=arg["type"],
                default=arg["default"],
            )

        # parse the arguments
        self.args = parser.parse_args()

    @staticmethod
    def year(x: str) -> int:
        """Validate the year CLI argument."""
        x = int(x)
        if x < 2018:
            raise argparse.ArgumentTypeError(
                f"Year must be greater than or equal to 2018, got {x}."
            )
        elif x > datetime.datetime.now().year:
            raise argparse.ArgumentTypeError(
                f"Year can not be in the future, got {x}."
            )
        return x

    @staticmethod
    def log_level(x: str) -> int:
        """Validate the log level argument."""
        match x:
            case "debug":
                return logging.DEBUG
            case "info":
                return logging.INFO
            case "warning":
                return logging.WARNING
            case "error":
                return logging.ERROR
            case "critical":
                return logging.CRITICAL
            case _:
                raise argparse.ArgumentTypeError(
                    f"Invalid log_level value, got {x}. Must be one of "
                    "'debug', 'info', 'warning', 'error', or 'critical'."
                )


def setup_logger(
    logger_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Build a logger instance.

    Parameters
    ----------
    logger_name : str
        name of logger.
    level : int, optional
        logger level (e.g., logging.DEBUG, logging.WARNING etc.), by default
        logging.INFO.

    Returns
    -------
    logging.Logger
        a logger instance with the requested properties.

    """
    # set up the logger and logging level
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # fix the logger format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up a stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)

    return logger


def mode(x):
    """Calculate the modal occurance."""
    return x.mode().iloc[0] if not x.mode().empty else None


# silence fastf1 noisy logs
LoggingManager().set_level(logging.ERROR)

# fastest lap points schema
points_schema = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


def main(args: argparse.Namespace) -> None:
    """Calculate fastest lap standings."""
    events = ff1.get_event_schedule(args.year, include_testing=False)
    results = []  # store race results
    num_races = 0  # number of races complete record
    logger = setup_logger("analyse-f1-fastest-lap-standings", args.log_level)
    for race_num in events.RoundNumber:
        # get the session and check that it's run
        session = ff1.get_session(args.year, race_num, "R")
        if session.event.EventDate > datetime.datetime.now():
            logger.debug(
                f"Skipping {session.event.EventName} as it's not happened yet."
            )
            continue

        # get the session laps
        logger.debug(f"Working on {session.event.EventName}...")
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
        valid_laps.loc[:, "LapTimeS"] = valid_laps[
            "LapTime"
        ].dt.total_seconds()
        valid_laps.loc[:, "IsValidSectors"] = np.isclose(
            valid_laps["Sector1S"]
            + valid_laps["Sector2S"]
            + valid_laps["Sector3S"],
            valid_laps["LapTimeS"],
            rtol=1e-03,
        )
        valid_laps = valid_laps[valid_laps.IsValidSectors].reset_index(
            drop=True
        )

        # calculate the fastest laps by driver
        fastest_laps = (
            valid_laps.groupby("Driver", as_index=False)
            .LapTimeS.min()
            .sort_values("LapTimeS", ascending=True)
            .rename(columns={"LapTimeS": "FastestLapTime"})
            .reset_index(drop=True)
        )

        # join on session metadata to fastest laps
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
        # make driver column name consistent for merging
        metadata = metadata.rename(columns={"Abbreviation": "Driver"})

        # construct a DSQ-ed identifier column
        metadata.loc[:, "DSQ"] = metadata.ClassifiedPosition.apply(
            lambda x: True if x == "D" else False
        )

        # merge thensort fastest laps, catering for the DSQ-ed drivers separately
        fastest_laps = fastest_laps.merge(metadata, on="Driver")
        fastest_laps.sort_values(
            ["DSQ", "FastestLapTime"], ascending=[True, True], inplace=True
        )

        # add position, points scored, and event name
        fastest_laps["Position"] = np.arange(1, len(fastest_laps) + 1)
        fastest_laps["FastestLapPoints"] = fastest_laps.Position.map(
            points_schema
        ).fillna(0)
        fastest_laps.loc[:, "EventName"] = session.event.EventName

        # log for later use during plotting
        last_race = session.event.EventName
        last_race_date = session.event.EventDate.strftime("%Y-%m-%d")
        num_races += 1
        logger.debug(
            f"{session.event.EventName} complete. P1: "
            f"{fastest_laps.iloc[0].Driver}"
        )

        results.append(fastest_laps)

    # combine all the results
    results = pd.concat(results)

    # calculate the drivers standings
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

    # calculate the constructors standings
    constructors_standings = (
        results.groupby("TeamName", as_index=False)
        .agg({"FastestLapPoints": "sum", "TeamColor": mode})
        .sort_values("FastestLapPoints", ascending=False)
        .reset_index(drop=True)
    )

    for name_col, standings in (
        ("Driver", drivers_standings),
        ("TeamName", constructors_standings),
    ):
        countback = (
            results.groupby(name_col, as_index=False)["Position"]
            .value_counts()
            .rename(columns={"count": "PositionAchievements"})
            .pivot(
                columns="Position",
                index=name_col,
                values="PositionAchievements",
            )
            .fillna(0)
        )
        # zfill to set valid sort order
        countback.columns = [
            "Position" + str(col).zfill(2)
            for col in countback.columns.get_level_values(0)
        ]
        countback.reset_index(inplace=True)

        # merge countback and sort standing by points and countback occurances
        standings = standings.merge(countback, on=name_col)
        countback_columns = list(countback.columns)
        countback_columns.remove(name_col)
        standings.sort_values(
            ["FastestLapPoints"] + sorted(countback_columns),
            inplace=True,
            ascending=False,
        )
        standings.reset_index(drop=True, inplace=True)

    # plot standings
    for plot_type, standings in (
        ("drivers", drivers_standings),
        ("constructors", constructors_standings),
    ):
        fig = grid_plot(
            standings,
            plot_type,
            "FastestLapPoints",
            f"{args.year} {plot_type.capitalize()} Fastest Lap Championship",
            f"Standings after {num_races} rounds (post {last_race} on "
            f"{last_race_date}).",
        )
        output_path = (
            f"./outputs/{args.year}_{plot_type}_fastest_laps_standings_"
            f"{num_races}_rounds.png"
        )
        fig.write_image(output_path)
        logger.info(
            f"{args.year} {plot_type} fastest lap plot saved to: {output_path}"
        )


if __name__ == "__main__":
    # calculate fastest laps standings
    main(CLI().args)
