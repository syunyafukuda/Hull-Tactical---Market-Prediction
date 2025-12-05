"""Unit tests for SU9 sweep_oof script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.feature_generation.su9.sweep_oof import (
    _count_su9_features,
    _parse_bool_grid,
    build_param_grid,
    parse_args,
)


class TestParseBoolGrid:
    """_parse_bool_grid のテスト。"""

    def test_none_returns_default(self) -> None:
        """None の場合はデフォルト値を返す。"""
        result = _parse_bool_grid(None)
        assert result == [True, False]

    def test_none_with_custom_default(self) -> None:
        """None とカスタムデフォルトの場合。"""
        result = _parse_bool_grid(None, default=[True])
        assert result == [True]

    def test_true_string(self) -> None:
        """'true' 文字列を True に変換。"""
        result = _parse_bool_grid(["true"])
        assert result == [True]

    def test_false_string(self) -> None:
        """'false' 文字列を False に変換。"""
        result = _parse_bool_grid(["false"])
        assert result == [False]

    def test_mixed_strings(self) -> None:
        """混合文字列のテスト。"""
        result = _parse_bool_grid(["true", "false"])
        assert result == [True, False]

    def test_case_insensitive(self) -> None:
        """大文字小文字を区別しない。"""
        result = _parse_bool_grid(["TRUE", "False", "TrUe"])
        assert result == [True, False, True]


class TestBuildParamGrid:
    """build_param_grid のテスト。"""

    def test_default_grid_size(self) -> None:
        """デフォルトグリッドは 64 (2^6) 通り。"""
        args = parse_args([])
        grid = build_param_grid(args)
        assert len(grid) == 64

    def test_skip_all_false(self) -> None:
        """--skip-all-false オプションで全 False を除外。"""
        args = parse_args(["--skip-all-false"])
        grid = build_param_grid(args)
        assert len(grid) == 63  # 64 - 1

        # 全 False の設定がないことを確認
        for config in grid:
            assert any(config.values()), "All False configuration should be excluded"

    def test_single_flag_grid(self) -> None:
        """1つのフラグのみ変更するグリッド。"""
        args = parse_args([
            "--include-dow-grid", "true",
            "--include-dom-grid", "true",
            "--include-month-grid", "true",
            "--include-month-flags-grid", "true",
            "--include-holiday-grid", "true",
            "--include-year-position-grid", "true", "false",
        ])
        grid = build_param_grid(args)
        assert len(grid) == 2  # year_position のみ 2通り

    def test_all_true_configuration_exists(self) -> None:
        """全て True の設定が含まれること。"""
        args = parse_args([])
        grid = build_param_grid(args)

        all_true = {
            "include_dow": True,
            "include_dom": True,
            "include_month": True,
            "include_month_flags": True,
            "include_holiday": True,
            "include_year_position": True,
        }
        assert all_true in grid

    def test_all_false_configuration_exists(self) -> None:
        """全て False の設定が含まれること（skip-all-false なし）。"""
        args = parse_args([])
        grid = build_param_grid(args)

        all_false = {
            "include_dow": False,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert all_false in grid


class TestCountSu9Features:
    """_count_su9_features のテスト。"""

    def test_all_true(self) -> None:
        """全て True の場合は 32 列。"""
        config = {
            "include_dow": True,
            "include_dom": True,
            "include_month": True,
            "include_month_flags": True,
            "include_holiday": True,
            "include_year_position": True,
        }
        assert _count_su9_features(config) == 32

    def test_all_false(self) -> None:
        """全て False の場合は 0 列。"""
        config = {
            "include_dow": False,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 0

    def test_dow_only(self) -> None:
        """曜日のみの場合は 7 列。"""
        config = {
            "include_dow": True,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 7

    def test_dom_only(self) -> None:
        """月内位置のみの場合は 3 列。"""
        config = {
            "include_dow": False,
            "include_dom": True,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 3

    def test_month_only(self) -> None:
        """月のみの場合は 12 列。"""
        config = {
            "include_dow": False,
            "include_dom": False,
            "include_month": True,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 12

    def test_month_flags_only(self) -> None:
        """月末・期末フラグのみの場合は 4 列。"""
        config = {
            "include_dow": False,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": True,
            "include_holiday": False,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 4

    def test_holiday_only(self) -> None:
        """祝日のみの場合は 4 列。"""
        config = {
            "include_dow": False,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": True,
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 4

    def test_year_position_only(self) -> None:
        """年内ポジションのみの場合は 2 列。"""
        config = {
            "include_dow": False,
            "include_dom": False,
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": False,
            "include_year_position": True,
        }
        assert _count_su9_features(config) == 2

    def test_partial_combination(self) -> None:
        """一部の組み合わせ。"""
        config = {
            "include_dow": True,  # 7
            "include_dom": True,  # 3
            "include_month": False,
            "include_month_flags": False,
            "include_holiday": True,  # 4
            "include_year_position": False,
        }
        assert _count_su9_features(config) == 14


class TestParseArgs:
    """parse_args のテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値のテスト。"""
        args = parse_args([])
        assert args.config_path == "configs/feature_generation.yaml"
        assert args.preprocess_config == "configs/preprocess.yaml"
        assert args.data_dir == "data/raw"
        assert args.out_dir == "results/ablation/SU9"
        assert args.n_splits == 5
        assert args.target_col == "market_forward_excess_returns"
        assert args.id_col == "date_id"

    def test_custom_out_dir(self) -> None:
        """カスタム出力ディレクトリ。"""
        args = parse_args(["--out-dir", "/tmp/custom_dir"])
        assert args.out_dir == "/tmp/custom_dir"

    def test_n_splits(self) -> None:
        """fold 数のテスト。"""
        args = parse_args(["--n-splits", "3"])
        assert args.n_splits == 3

    def test_grid_params(self) -> None:
        """グリッドパラメータのテスト。"""
        args = parse_args([
            "--include-dow-grid", "true",
            "--include-dom-grid", "false",
        ])
        assert args.include_dow_grid == ["true"]
        assert args.include_dom_grid == ["false"]

    def test_skip_all_false_flag(self) -> None:
        """--skip-all-false フラグのテスト。"""
        args = parse_args(["--skip-all-false"])
        assert args.skip_all_false is True

        args_no_skip = parse_args([])
        assert args_no_skip.skip_all_false is False

    def test_model_params(self) -> None:
        """モデルパラメータのテスト。"""
        args = parse_args([
            "--learning-rate", "0.1",
            "--n-estimators", "100",
            "--num-leaves", "31",
        ])
        assert args.learning_rate == 0.1
        assert args.n_estimators == 100
        assert args.num_leaves == 31


class TestGridCartesianProduct:
    """グリッドのデカルト積が正しいことのテスト。"""

    def test_cartesian_product_correctness(self) -> None:
        """デカルト積の正確性。"""
        args = parse_args([
            "--include-dow-grid", "true", "false",
            "--include-dom-grid", "true",
            "--include-month-grid", "true",
            "--include-month-flags-grid", "true",
            "--include-holiday-grid", "true",
            "--include-year-position-grid", "true",
        ])
        grid = build_param_grid(args)

        # dow のみ 2 通り、他は 1 通り
        assert len(grid) == 2

        # 両方の dow 設定が含まれる
        dow_values = [config["include_dow"] for config in grid]
        assert True in dow_values
        assert False in dow_values

    def test_all_configs_have_required_keys(self) -> None:
        """全ての設定が必要なキーを持つこと。"""
        args = parse_args([])
        grid = build_param_grid(args)

        required_keys = {
            "include_dow",
            "include_dom",
            "include_month",
            "include_month_flags",
            "include_holiday",
            "include_year_position",
        }

        for config in grid:
            assert set(config.keys()) == required_keys

    def test_all_values_are_bool(self) -> None:
        """全ての値がブール型であること。"""
        args = parse_args([])
        grid = build_param_grid(args)

        for config in grid:
            for value in config.values():
                assert isinstance(value, bool)
