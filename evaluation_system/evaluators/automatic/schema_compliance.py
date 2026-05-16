"""Schema compliance checks for tool-call arguments.

The repository carries ``schema_compliance_schema.yaml`` generated from the MCP
runtime. To keep the evaluation pipeline dependency-light, this module parses
only the subset needed for scoring:

- servers / tools
- canonical_input_schema.properties
- canonical_input_schema.required
- JSON-schema-ish ``type`` / ``anyOf`` / array ``items``

Semantic hints in the YAML are intentionally not scored here. Schema compliance
is structural: tool exists, arguments are an object, required arguments are
present, there are no unknown arguments, and values have compatible JSON types.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PropertySpec:
    types: frozenset[str]
    item_types: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolSchema:
    server: str
    tool: str
    required: frozenset[str]
    properties: dict[str, PropertySpec]


@dataclass(frozen=True)
class SchemaValidationResult:
    tool_exists: bool
    schema_valid: bool
    reason: str = ""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_comment_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _schema_path() -> Path:
    # evaluation_system/evaluators/automatic/schema_compliance.py -> repo root
    return Path(__file__).resolve().parents[3] / "schema_compliance_schema.yaml"


def _block_until(lines: list[str], start: int, *, max_indent: int) -> tuple[list[str], int]:
    out: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.strip() and _indent(line) <= max_indent:
            break
        out.append(line)
        i += 1
    return out, i


def _parse_required(block: list[str]) -> frozenset[str]:
    required: list[str] = []
    in_required = False
    required_indent = 0
    for line in block:
        stripped = line.strip()
        if not stripped:
            continue
        ind = _indent(line)
        if ind == 10 and stripped == "required:":
            in_required = True
            required_indent = ind
            continue
        if in_required:
            if ind <= required_indent and not stripped.startswith("- "):
                in_required = False
            elif stripped.startswith("- "):
                required.append(_strip_comment_value(stripped[2:]))
    return frozenset(required)


def _property_blocks(block: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    in_properties = False
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for line in block:
        stripped = line.strip()
        if not stripped:
            continue
        ind = _indent(line)
        if ind == 10 and stripped == "properties:":
            in_properties = True
            continue
        if not in_properties:
            continue
        if ind <= 10:
            break
        if ind == 12 and stripped.endswith(":") and not stripped.startswith("- "):
            if current_name is not None:
                out[current_name] = current_lines
            current_name = stripped[:-1]
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        out[current_name] = current_lines
    return out


def _parse_property(lines: list[str]) -> PropertySpec:
    types: set[str] = set()
    item_types: set[str] = set()
    pending_array_item = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- type:"):
            types.add(_strip_comment_value(stripped.split(":", 1)[1]))
            continue
        if stripped == "- items:":
            pending_array_item = True
            continue
        if stripped == "items:":
            pending_array_item = True
            continue
        if stripped.startswith("type:"):
            value = _strip_comment_value(stripped.split(":", 1)[1])
            if pending_array_item:
                item_types.add(value)
                pending_array_item = False
            else:
                types.add(value)
            continue
    return PropertySpec(types=frozenset(types), item_types=frozenset(item_types))


def _parse_schema_yaml(path: Path) -> dict[tuple[str, str], ToolSchema]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    schemas: dict[tuple[str, str], ToolSchema] = {}
    server: Optional[str] = None
    tool: Optional[str] = None
    in_servers = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        ind = _indent(line)

        if ind == 0 and stripped == "servers:":
            in_servers = True
            i += 1
            continue
        if not in_servers:
            i += 1
            continue
        if ind == 2 and stripped.endswith(":"):
            server = stripped[:-1]
            tool = None
            i += 1
            continue
        if server and ind == 6 and stripped.endswith(":"):
            tool = stripped[:-1]
            i += 1
            continue
        if server and tool and ind == 8 and stripped == "canonical_input_schema:":
            block, next_i = _block_until(lines, i + 1, max_indent=8)
            properties = {
                name: _parse_property(prop_lines)
                for name, prop_lines in _property_blocks(block).items()
            }
            schemas[(server, tool)] = ToolSchema(
                server=server,
                tool=tool,
                required=_parse_required(block),
                properties=properties,
            )
            i = next_i
            continue
        i += 1
    return schemas


def _json_type_ok(value: Any, allowed: frozenset[str], item_types: frozenset[str]) -> bool:
    if not allowed:
        return True
    if value is None:
        return "null" in allowed
    if "string" in allowed and isinstance(value, str):
        return True
    if "boolean" in allowed and isinstance(value, bool):
        return True
    if "number" in allowed and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if "integer" in allowed and isinstance(value, int) and not isinstance(value, bool):
        return True
    if "object" in allowed and isinstance(value, dict):
        return True
    if "array" in allowed and isinstance(value, list):
        if not item_types:
            return True
        return all(_json_type_ok(item, item_types, frozenset()) for item in value)
    return False


class SchemaComplianceValidator:
    def __init__(self, schemas: dict[tuple[str, str], ToolSchema]) -> None:
        self._schemas = schemas
        by_tool: dict[str, list[tuple[str, str]]] = {}
        for key in schemas:
            by_tool.setdefault(key[1], []).append(key)
        self._unique_tool_names = {
            tool: keys[0] for tool, keys in by_tool.items() if len(keys) == 1
        }

    def _resolve(self, tool_name: str) -> tuple[Optional[ToolSchema], str]:
        name = str(tool_name or "").strip()
        if "." in name:
            server, tool = name.split(".", 1)
            schema = self._schemas.get((server, tool))
            return schema, f"{server}.{tool}"
        key = self._unique_tool_names.get(name)
        if key is None:
            return None, name
        return self._schemas.get(key), f"{key[0]}.{key[1]}"

    def validate(self, tool_name: str, arguments: Any) -> SchemaValidationResult:
        schema, resolved_name = self._resolve(tool_name)
        if schema is None:
            return SchemaValidationResult(False, False, f"unknown tool: {resolved_name}")
        if not isinstance(arguments, dict):
            return SchemaValidationResult(True, False, "arguments is not an object")

        unknown = set(arguments) - set(schema.properties)
        if unknown:
            return SchemaValidationResult(True, False, f"unknown arguments: {sorted(unknown)}")

        for required in schema.required:
            if required not in arguments:
                return SchemaValidationResult(True, False, f"missing required argument: {required}")
            value = arguments.get(required)
            spec = schema.properties.get(required)
            if spec and "string" in spec.types and value == "":
                return SchemaValidationResult(True, False, f"empty required string: {required}")

        for name, value in arguments.items():
            spec = schema.properties.get(name)
            if spec is None:
                continue
            if not _json_type_ok(value, spec.types, spec.item_types):
                return SchemaValidationResult(True, False, f"type mismatch: {name}")

        return SchemaValidationResult(True, True, "")


@lru_cache(maxsize=1)
def default_schema_validator() -> SchemaComplianceValidator:
    return SchemaComplianceValidator(_parse_schema_yaml(_schema_path()))
