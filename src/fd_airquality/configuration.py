"""TOML defaults with explicit command-line arguments taking precedence."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import fields
from pathlib import Path

from fd_airquality.settings import RunSettings

DEFAULT_CONFIG = Path("fd-airquality.toml")
RUN_COMMANDS = {"simulate", "baseline", "benchmark"}


def _value(action: argparse.Action, value, location: str):
    multiple = action.nargs in {"+", "*"} or isinstance(action, argparse._AppendAction)
    if multiple:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{location}: se requiere una lista no vacía.")
        values = value
    else:
        values = [value]
    converted = []
    for item in values:
        kind = bool if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)) else action.type or str
        valid = (isinstance(item, (int, float)) and not isinstance(item, bool) if kind is float
                 else isinstance(item, str) if kind is Path else type(item) is kind)
        if not valid:
            raise ValueError(f"{location}: tipo inválido; se esperaba {kind.__name__}.")
        item = kind(item)
        if action.choices is not None and item not in action.choices:
            raise ValueError(f"{location}: {item!r} no está entre {list(action.choices)}.")
        converted.append(item)
    return converted if multiple else converted[0]


def _load(path: Path, commands: dict[str, argparse.ArgumentParser]) -> dict[str, dict]:
    with path.open("rb") as stream:
        try:
            document = tomllib.load(stream)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"TOML inválido en {path}: {exc}") from exc
    unknown = document.keys() - (commands.keys() | {"run"})
    if unknown:
        raise ValueError(f"Secciones desconocidas en {path}: {sorted(unknown)}")
    sections = {}
    for section, options in document.items():
        if not isinstance(options, dict):
            raise TypeError(f"{path}: [{section}] debe ser una tabla TOML.")
        command = commands["simulate" if section == "run" else section]
        actions = {action.dest: action for action in command._actions if action.dest not in {"help", "config"}}
        if section == "run":
            actions = {key: value for key, value in actions.items() if key in {field.name for field in fields(RunSettings)}}
        converted = {}
        for key, value in options.items():
            name = key.replace("-", "_")
            if name not in actions:
                raise ValueError(f"Opción desconocida en {path}: [{section}].{key}")
            if name in converted:
                raise ValueError(f"Opción duplicada en {path}: [{section}].{key}")
            converted[name] = _value(actions[name], value, f"{path}: [{section}].{key}")
        sections[section] = converted
    return sections


def parse_configured_args(parser: argparse.ArgumentParser, commands: dict[str, argparse.ArgumentParser],
                          argv: list[str] | None) -> argparse.Namespace:
    defaults = vars(parser.parse_args(argv))
    explicit_path = defaults.get("config")
    path = Path(explicit_path).expanduser() if explicit_path is not None else DEFAULT_CONFIG
    if not path.is_file():
        if explicit_path is not None:
            raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
        return argparse.Namespace(**defaults)
    try:
        sections = _load(path, commands)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    command = defaults["command"]
    configured = (sections.get("run", {}) if command in RUN_COMMANDS else {}) | sections.get(command, {})
    actions = [action for owner in (parser, *commands.values()) for action in owner._actions]
    saved = [action.default for action in actions]
    try:
        for action in actions:
            action.default = argparse.SUPPRESS
        explicit = vars(parser.parse_args(argv))
    finally:
        for action, default in zip(actions, saved, strict=True):
            action.default = default
    return argparse.Namespace(**(defaults | configured | explicit | {"config": path.resolve()}))
