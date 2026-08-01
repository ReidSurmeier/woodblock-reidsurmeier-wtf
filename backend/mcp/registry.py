"""Central MCP tool registry for the v23 server surface."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from backend.mcp.errors import ToolResult, WoodblockError
from backend.mcp.tools import calibration, carve, core, hitl, introspection, overlay, session


@dataclass(frozen=True)
class MCPTool:
    """A callable plus the MCP metadata needed for tools/list and tools/call."""

    name: str
    tier: str
    func: Callable[..., Any]
    param_aliases: dict[str, str] = field(default_factory=dict)

    @property
    def description(self) -> str:
        doc = inspect.getdoc(self.func) or self.name
        return doc.split("\n\n", 1)[0]

    @property
    def input_schema(self) -> dict[str, Any]:
        return _schema_for_callable(self.func)

    def call(self, arguments: dict[str, Any] | None) -> ToolResult[dict[str, Any]]:
        args = dict(arguments or {})
        for alias, canonical in self.param_aliases.items():
            if alias in args and canonical not in args:
                args[canonical] = args.pop(alias)
        try:
            inspect.signature(self.func).bind(**args)
        except TypeError as exc:
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="INVALID_TOOL_ARGUMENTS",
                        message=str(exc),
                        hint=f"check tools/list inputSchema for {self.name}",
                        recoverable=True,
                    )
                ],
            )
        result = self.func(**args)
        return _coerce_tool_result(result)


def _module_tools(module: Any, tier: str) -> list[MCPTool]:
    aliases = {"propose_stack": {"image_path": "path"}}
    return [
        MCPTool(
            name=name,
            tier=tier,
            func=getattr(module, name),
            param_aliases=aliases.get(name, {}),
        )
        for name in module.__all__
    ]


def _build_registry() -> dict[str, MCPTool]:
    modules = (
        (core, "core"),
        (hitl, "hitl"),
        (calibration, "calibration"),
        (introspection, "introspection"),
        (session, "session"),
        (carve, "carve"),
        (overlay, "overlay"),
    )
    tools: dict[str, MCPTool] = {}
    for module, tier in modules:
        for tool in _module_tools(module, tier):
            if tool.name in tools:
                raise RuntimeError(f"duplicate MCP tool registered: {tool.name}")
            tools[tool.name] = tool
    return tools


TOOLS: dict[str, MCPTool] = _build_registry()


def list_mcp_tools() -> list[dict[str, Any]]:
    """Return MCP tools/list metadata in stable tier/name order."""
    return [
        {
            "name": tool.name,
            "description": f"[{tool.tier}] {tool.description}",
            "inputSchema": tool.input_schema,
        }
        for tool in sorted(TOOLS.values(), key=lambda t: (t.tier, t.name))
    ]


def call_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> ToolResult[dict[str, Any]]:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="UNKNOWN_TOOL",
                    message=f"unknown tool: {name}",
                    hint="call tools/list for the registered woodblock_stack surface",
                    recoverable=True,
                )
            ],
        )
    return tool.call(arguments)


def tool_result_to_jsonable(result: Any) -> dict[str, Any]:
    return _coerce_tool_result(result).model_dump(mode="json")


def _coerce_tool_result(result: Any) -> ToolResult[dict[str, Any]]:
    if isinstance(result, ToolResult):
        return result
    if hasattr(result, "model_dump"):
        return ToolResult(ok=True, data=result.model_dump(mode="json"))
    if isinstance(result, dict):
        return ToolResult(ok=True, data=result)
    return ToolResult(ok=True, data={"value": result})


def _schema_for_callable(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        schema = _schema_for_annotation(annotation)
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        enum = list(args)
        types_seen = {type(v) for v in enum}
        schema_type = "string" if types_seen <= {str} else None
        out: dict[str, Any] = {"enum": enum}
        if schema_type:
            out["type"] = schema_type
        return out
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            schema = _schema_for_annotation(non_none[0])
            return {"anyOf": [schema, {"type": "null"}]}
        return {"anyOf": [_schema_for_annotation(arg) for arg in args]}
    if origin in (list, tuple, set):
        item_schema = _schema_for_annotation(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation in (dict,):
        return {"type": "object"}
    if annotation in (list, tuple, set):
        return {"type": "array"}
    return {}


__all__ = ["MCPTool", "TOOLS", "call_mcp_tool", "list_mcp_tools", "tool_result_to_jsonable"]
