"""Verify the script only publishes completed and validated JSON."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "examples/qwen35b-llama/action_items.py"
spec = importlib.util.spec_from_file_location("action_items", path)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class ActionItemsTests(unittest.TestCase):
    def invoke(self, body, stream=False):
        argv = ["action_items.py", "--json"]
        if stream:
            argv.append("--stream")
        with patch.object(script, "urlopen", return_value=io.BytesIO(body)) as request, \
                patch("sys.argv", argv), patch("sys.stdin", io.StringIO("Lan gửi báo cáo.")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO):
            code = script.main()
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertNotIn("temperature", sent)
        return code, stdout.getvalue()

    def test_json_validates_null_deadline_and_empty_list(self):
        for expected in ({"items": []}, {"items": [{"owner": "Lan", "task": "gửi báo cáo", "due": None}]}):
            choice = {"message": {"content": json.dumps(expected)}, "finish_reason": "stop"}
            code, stdout = self.invoke(json.dumps({"choices": [choice]}).encode())
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), expected)

    def test_stream_publishes_nothing_for_truncation_disconnect_or_invalid_fields(self):
        for content, finish, done in (('{"items":[]}', "length", True),
                                      ('{"items":[]}', "stop", False),
                                      ('{"items":[{"owner":"Lan"}]}', "stop", True)):
            with self.subTest(finish=finish, done=done, content=content):
                choice = {"delta": {"content": content}, "finish_reason": finish}
                raw = b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n"
                if done:
                    raw += b"data: [DONE]\n\n"
                self.assertEqual(self.invoke(raw, stream=True), (1, ""))

    def test_valid_stream_publishes_completed_object(self):
        chunks = ({"delta": {"content": '{"items":'}}, {"delta": {"content": "[]}"}},
                  {"delta": {}, "finish_reason": "stop"})
        raw = b"".join(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n" for choice in chunks)
        code, stdout = self.invoke(raw + b"data: [DONE]\n\n", stream=True)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), {"items": []})
