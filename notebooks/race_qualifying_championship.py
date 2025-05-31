"""Build a qualifying standings visualisation."""

import argparse
import datetime
import logging
import sys

import fastf1 as ff1
import numpy as np
import pandas as pd
from fastf1.logger import LoggingManager
from plot import grid_plot

# %%

# %%


class CLI:
    """Handle the cli inputs."""

    def __init__(self):
        """Parse the command line arguments."""
        parser = argparse.ArgumentParser(
            description=(
                "Build a F1 qualifying championship standings for drivers and"
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
            },
            {
                "name": "session",
                "help": (
                    "The session type, must be either 'qualifying' or 'race'."
                ),
                "type": self.session,
            },
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

    @staticmethod
    def session(x: str) -> str:
        """Validate session argument."""
        if (x != "qualifying") and (x != "race"):
            raise argparse.ArgumentTypeError(
                f"Session must be either 'qualifying' or 'race', got {x}."
            )
        return x


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

# qualifying points schema
race_points_schema = {
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
sprint_2021_schema = {
    1: 3,
    2: 2,
    3: 1,
}
sprint_points_schema = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1,
}


def main(args: argparse.Namespace) -> None:  # noqa:C901
    """Calculate qualifying standings."""
    events = ff1.get_event_schedule(args.year, include_testing=False)
    results = []  # store race results
    rounds_completed = 0  # number of races complete record
    logger = setup_logger("analyse-f1-fastest-lap-standings", args.log_level)
    points_type = (
        "QualiPoints" if args.session == "qualifying" else "RacePoints"
    )
    title_snippet = "Qualifying " if args.session == "qualifying" else ""
    for race_num, event_name, event_format in zip(
        events.RoundNumber, events.EventName, events.EventFormat
    ):
        if args.session == "qualifying":
            if args.year < 2021:
                session_ids = ["Qualifying"]
                points_schemas = [race_points_schema]
            elif args.year == 2021:
                session_ids = ["Qualifying"]
                if event_format == "sprint":
                    points_schemas = [sprint_2021_schema]
                else:
                    points_schemas = [race_points_schema]
            elif args.year == 2022:
                session_ids = ["Qualifying"]
                if event_format == "sprint":
                    points_schemas = [sprint_points_schema]
                else:
                    points_schemas = [race_points_schema]
            elif args.year == 2023:
                session_ids = ["Qualifying", "Sprint Shootout"]
                points_schemas = [race_points_schema, sprint_points_schema]
            else:
                session_ids = ["Sprint Qualifying", "Qualifying"]
                points_schemas = [sprint_points_schema, race_points_schema]
        else:
            if args.year < 2021:
                session_ids = ["Race"]
                points_schemas = [race_points_schema]
            elif args.year == 2021:
                session_ids = ["Sprint", "Race"]
                points_schemas = [sprint_2021_schema, race_points_schema]
            else:
                session_ids = ["Sprint", "Race"]
                points_schemas = [sprint_points_schema, race_points_schema]

        for session_id, points_schema in zip(session_ids, points_schemas):
            # get the session and check that it's run
            try:
                session = ff1.get_session(args.year, race_num, session_id)
            except ValueError as e:
                if (
                    f"Session type '{session_id}' does not exist for this "
                    "event" in str(e)
                ):
                    logger.debug(
                        f"Skipping {event_name}, race {race_num}, {session_id}"
                        ", because the session does not exist..."
                    )
                    continue
                else:
                    raise e
            if session.event.EventDate > datetime.datetime.now():
                logger.debug(
                    f"Skipping {session.event.EventName}, race {race_num}, "
                    f"{session_id}, because it has not happened yet."
                )
                continue
            logger.debug(
                f"Working on {session.event.EventName}, race {race_num}, "
                f"{session_id}..."
            )

            # get the session laps - need laps for sprint quali
            session.load(telemetry=False, laps=True, weather=False)

            # get session results
            session_results = session.results[
                [
                    "Abbreviation",
                    "BroadcastName",
                    "DriverNumber",
                    "TeamName",
                    "TeamColor",
                    "Position",
                    "ClassifiedPosition",
                ]
            ]
            # make driver column name consistent for merging
            session_results = session_results.rename(
                columns={"Abbreviation": "Driver"}
            )

            # add position, points scored, and event name
            session_results["Position"] = np.arange(
                1, len(session_results) + 1
            )
            if event_name == "Belgian Grand Prix" and args.year == 2021:
                points_schema = {
                    pos: 0.5 * pts for pos, pts in points_schema.items()
                }
            session_results[points_type] = session_results.Position.map(
                points_schema
            ).fillna(0)
            session_results.loc[:, "EventName"] = session.event.EventName
            session_results.loc[:, "SessionType"] = session_id

            # log for later use during plotting
            last_race = session.event.EventName
            last_race_date = session.event.EventDate.strftime("%Y-%m-%d")

            logger.debug(
                f"{session.event.EventName}, race {race_num}, {session_id}, "
                f"complete. P1: {session_results.iloc[0].Driver}"
            )

            if (
                args.year >= 2019
                and args.year <= 2024
                and session_id == "Race"
            ):
                # add extra point for fastest lap
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
                valid_laps = valid_laps.reset_index(drop=True)

                # calculate the fastest laps by driver
                fastest_laps = (
                    valid_laps.groupby("Driver", as_index=False)
                    .LapTimeS.min()
                    .sort_values("LapTimeS", ascending=True)
                    .rename(columns={"LapTimeS": "FastestLapTime"})
                    .reset_index(drop=True)
                )

                # arrange extra point if needed
                try:
                    fastest_driver = fastest_laps.iloc[0].Driver
                    if (
                        fastest_driver
                        in session_results.iloc[0:10].Driver.to_list()
                    ):
                        session_results.loc[
                            session_results["Driver"] == fastest_driver,
                            points_type,
                        ] += 1
                        logger.debug(
                            f"Adding 1pt for {fastest_driver} fastest lap"
                        )
                    else:
                        logger.debug(
                            f"{fastest_driver} not in the top 10, so no "
                            "fastest lap point."
                        )
                except IndexError as e:
                    if args.year == 2021 and race_num == 12:
                        logger.debug("No fastest lap at 2021 Belgium.")
                    else:
                        raise e

            results.append(session_results)

            # set to latest race number
            rounds_completed = race_num

    # combine all the results
    results = pd.concat(results)

    # calculate the drivers standings
    drivers_standings = (
        results.groupby("Driver", as_index=False)
        .agg(
            {
                points_type: "sum",
                "BroadcastName": mode,
                "DriverNumber": mode,
                "TeamName": mode,
                "TeamColor": mode,
            }
        )
        .sort_values(points_type, ascending=False)
        .reset_index(drop=True)
    )

    # calculate the constructors standings
    constructors_standings = (
        results.groupby("TeamName", as_index=False)
        .agg({points_type: "sum", "TeamColor": mode})
        .sort_values(points_type, ascending=False)
        .reset_index(drop=True)
    )

    for name_col, plot_type, standings in (
        ("Driver", "drivers", drivers_standings),
        ("TeamName", "constructors", constructors_standings),
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
        standings = standings.sort_values(
            [points_type] + sorted(countback_columns),
            ascending=False,
        )
        standings = standings.reset_index(drop=True)

        fig = grid_plot(
            standings,
            plot_type,
            points_type,
            f"{args.year} {plot_type.capitalize()} {title_snippet}"
            "Championship",
            f"Standings after {rounds_completed} rounds (post {last_race} on "
            f"{last_race_date})",
        )
        output_path = (
            f"./outputs/{args.year}_{plot_type}_{args.session}_standings_"
            f"{rounds_completed}_rounds.png"
        )
        fig.write_image(output_path)
        logger.info(
            f"{args.year} {plot_type} qualifying plot saved to: {output_path}"
        )


if __name__ == "__main__":
    # calculate qualifying championship standings
    main(CLI().args)
