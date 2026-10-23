from __future__ import annotations

import pytest

from forge.models import Message, Role, ToolCall, ToolDefinition
from forge.models.protocols import (
    extract_json,
    parse_text_tool_calls,
    render_tool_instructions,
    split_reasoning,
    to_prompt_protocol,
    with_system_suffix,
)

READ = ToolDefinition(
    name="read_file",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)
TOOLS = [READ, ToolDefinition(name="list_dir", description="List a directory")]
WRITE = ToolDefinition(
    name="write_file",
    description="Write a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_parents": {"type": "boolean"},
        },
    },
)


class TestSplitReasoning:
    def test_no_reasoning(self) -> None:
        assert split_reasoning("hello") == ("hello", None)

    def test_closed_block(self) -> None:
        answer, reasoning = split_reasoning("<think>step 1\nstep 2</think>\n\nThe answer is 4.")
        assert answer == "The answer is 4."
        assert reasoning == "step 1\nstep 2"

    def test_unclosed_block_is_all_reasoning(self) -> None:
        assert split_reasoning("<think>still thinking...") == ("", "still thinking...")

    def test_orphan_closing_tag(self) -> None:
        # Templates that put <think> in the prompt leave only the closing tag.
        assert split_reasoning("pondering</think>Final") == ("Final", "pondering")


class TestTextToolCalls:
    def test_single_call(self) -> None:
        text = (
            "Let me look.\n<tool_call>\n"
            '{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        )
        remaining, calls = parse_text_tool_calls(text, TOOLS)
        assert remaining == "Let me look."
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a.py"}
        assert calls[0].parse_error is None

    def test_multiple_calls_get_distinct_ids(self) -> None:
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>'
            '<tool_call>{"name": "list_dir", "arguments": {}}</tool_call>'
        )
        _, calls = parse_text_tool_calls(text, TOOLS)
        assert [c.name for c in calls] == ["read_file", "list_dir"]
        assert len({c.id for c in calls}) == 2

    def test_fenced_body_and_unclosed_final_block(self) -> None:
        text = '<tool_call>\n```json\n{"name": "list_dir", "arguments": {}}\n```'
        _, calls = parse_text_tool_calls(text, TOOLS)
        assert [c.name for c in calls] == ["list_dir"]
        assert calls[0].parse_error is None

    def test_unknown_tool_is_reported_not_dropped(self) -> None:
        _, calls = parse_text_tool_calls(
            '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', TOOLS
        )
        assert calls[0].name == "rm_rf"
        assert calls[0].parse_error is not None
        assert "unknown tool" in calls[0].parse_error

    def test_invalid_json_is_reported(self) -> None:
        _, calls = parse_text_tool_calls("<tool_call>{name: read_file}</tool_call>", TOOLS)
        assert calls[0].parse_error is not None

    def test_string_encoded_arguments(self) -> None:
        _, calls = parse_text_tool_calls(
            '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"x\\"}"}</tool_call>',
            TOOLS,
        )
        assert calls[0].arguments == {"path": "x"}

    def test_qwen_nemotron_xml_tool_call(self) -> None:
        text = (
            "Writing the app now.\n"
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>src/App.tsx</parameter>\n"
            "<parameter=content>\n"
            "export default function App() {\n"
            "  return <h1>KIRTAN SHAH</h1>;\n"
            "}\n"
            "</parameter>\n"
            "<parameter=create_parents>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        remaining, calls = parse_text_tool_calls(text, [*TOOLS, WRITE])

        assert remaining == "Writing the app now."
        assert len(calls) == 1
        assert calls[0].name == "write_file"
        assert calls[0].parse_error is None
        assert calls[0].arguments["path"] == "src/App.tsx"
        assert calls[0].arguments["create_parents"] is True
        assert "export default function App()" in calls[0].arguments["content"]
        assert "<h1>KIRTAN SHAH</h1>" in calls[0].arguments["content"]

    def test_xml_tool_call_unknown_tool_is_reported(self) -> None:
        _, calls = parse_text_tool_calls(
            "<tool_call><function=rm_rf><parameter=path>/</parameter></function></tool_call>",
            TOOLS,
        )
        assert calls[0].name == "rm_rf"
        assert calls[0].arguments == {"path": "/"}
        assert calls[0].parse_error is not None
        assert "unknown tool" in calls[0].parse_error

    def test_xml_tool_call_with_unclosed_tags(self) -> None:
        _, calls = parse_text_tool_calls(
            "<tool_call><function=read_file><parameter=path>a.py</parameter>",
            TOOLS,
        )
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a.py"}
        assert calls[0].parse_error is None

    def test_bare_json_fallback_only_for_known_tools(self) -> None:
        _, calls = parse_text_tool_calls(
            '```json\n{"name": "read_file", "arguments": {"path": "b"}}\n```', TOOLS
        )
        assert calls[0].arguments == {"path": "b"}
        _, none = parse_text_tool_calls('{"name": "someone", "arguments": {}}', TOOLS)
        assert none == []

    def test_malformed_call_with_dangling_close_tag(self) -> None:
        # Verbatim shape of a qwen2.5:7b reply observed in a live run.
        text = ' Ronaldo\n{"name": "list_dir", "arguments": {}}\n</tool_call>'
        remaining, calls = parse_text_tool_calls(text, TOOLS)
        assert [c.name for c in calls] == ["list_dir"]
        assert calls[0].parse_error is None
        assert remaining == "Ronaldo"

    def test_json_answers_are_not_mistaken_for_calls(self) -> None:
        text = 'The config is {"name": "prod", "arguments": []} as expected.'
        assert parse_text_tool_calls(text, TOOLS)[1] == []

    def test_plain_answer_has_no_calls(self) -> None:
        assert parse_text_tool_calls("The file is fine.", TOOLS) == ("The file is fine.", [])

    def test_instructions_reflect_tool_choice(self) -> None:
        assert "MUST call the `read_file`" in render_tool_instructions(TOOLS, "read_file")
        assert "MUST call at least one" in render_tool_instructions(TOOLS, "required")
        auto = render_tool_instructions(TOOLS, "auto")
        assert "read_file" in auto
        assert "list_dir" in auto


class TestPromptProtocolHistory:
    def test_converts_and_merges(self) -> None:
        call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
        history = [
            Message.system("sys"),
            Message.user("read a"),
            Message.assistant("Reading.", [call]),
            Message.tool_result(call, "contents"),
            Message.user("and now?"),
        ]
        converted = to_prompt_protocol(history)
        assert [m.role for m in converted] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
        assert "<tool_call>" in converted[2].text
        assert converted[2].tool_calls == []
        # tool result and following user turn merged into one user message
        assert '<tool_result name="read_file" id="c1">' in converted[3].text
        assert converted[3].text.endswith("and now?")

    def test_with_system_suffix(self) -> None:
        assert with_system_suffix([Message.user("hi")], "S")[0] == Message.system("S")
        merged = with_system_suffix([Message.system("A"), Message.user("hi")], "B")
        assert merged[0].text == "A\n\nB"
        assert len(merged) == 2


class TestExtractJson:
    @pytest.mark.parametrize(
        "text",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            'Sure! Here it is:\n{"a": 1}\nHope that helps.',
            '<think>{"wrong": true} hmm</think>{"a": 1}',
        ],
    )
    def test_lenient_extraction(self, text: str) -> None:
        assert extract_json(text) == {"a": 1}

    def test_braces_inside_strings(self) -> None:
        assert extract_json('note: {"code": "if (x) { y }", "n": 2}') == {
            "code": "if (x) { y }",
            "n": 2,
        }

    def test_skips_unparseable_spans(self) -> None:
        assert extract_json('{not json} then {"ok": true}') == {"ok": True}

    def test_array(self) -> None:
        assert extract_json("[1, 2]") == [1, 2]

    def test_failure(self) -> None:
        with pytest.raises(ValueError):
            extract_json("no json here")
