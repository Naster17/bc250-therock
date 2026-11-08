#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ensure busy/unknown daemon state refuses before any GPU work."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc250"))
import board_preflight as board


class BoardPreflight(unittest.TestCase):
    def test_nonroot_refused(self):
        with mock.patch.object(board.os, "geteuid", return_value=1000):
            with self.assertRaises(AssertionError):
                board.preflight("nonroot")

    def test_busy_docker_refused(self):
        with mock.patch.object(board.os, "geteuid", return_value=0), mock.patch.object(
            board, "docker_containers", return_value=["foreign"]
        ), mock.patch.object(board.subprocess, "check_output") as query:
            with self.assertRaises(AssertionError):
                board.preflight("busy")
            query.assert_not_called()

    def test_unknown_docker_refused(self):
        with mock.patch.object(board.os, "geteuid", return_value=0), mock.patch.object(
            board, "docker_containers", side_effect=TimeoutError("unavailable")
        ):
            with self.assertRaises(TimeoutError):
                board.preflight("unknown")

    def test_busy_podman_refused(self):
        with mock.patch.object(board.os, "geteuid", return_value=0), mock.patch.object(
            board, "docker_containers", return_value=[]
        ), mock.patch.object(
            board.subprocess, "check_output", return_value="foreign\n"
        ):
            with self.assertRaises(AssertionError):
                board.preflight("podman")
