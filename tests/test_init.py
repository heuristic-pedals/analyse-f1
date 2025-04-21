"""Top level package tests."""

import re

import analyse_f1


class TestAnalyseF1:
    """Top level tests for the analyse_f1 package name and version number."""

    def test_package_name(self) -> None:
        """Test package name."""
        expected_name = "analyse_f1"
        package_name = analyse_f1.__name__
        assert package_name == expected_name, (
            f"Package name is not `{expected_name}`. Got {package_name}."
        )

    def test_analyse_f1_version(self) -> None:
        """Test version attribute is accessible."""
        # regex expression taken from https://semver.org/#is-there-a-suggested-
        # regular-expression-regex-to-check-a-semver-string
        pattern = (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*["
            r"a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-"
            r"]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        version = analyse_f1.__version__
        assert re.search(pattern, version), (
            "Package version does not follow expected semvar format. Got "
            f"{version}"
        )
