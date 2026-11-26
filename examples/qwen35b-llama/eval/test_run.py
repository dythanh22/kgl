import io
import json
import unittest
from urllib.error import HTTPError

from run import check, complete


class FakeOpener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        if isinstance(self.response, Exception):
            raise self.response
        return io.BytesIO(self.response)


def sse(*events):
    return b"".join(
        b"data: " + (b"[DONE]" if event == "[DONE]" else json.dumps(event).encode()) + b"\n\n"
        for event in events
    )


class EvaluationTests(unittest.TestCase):
    def test_stream_records_reasoning_content_usage_and_done(self):
        stream = sse(
            {"choices": [{"delta": {"reasoning_content": "hidden"}}]},
            {"choices": [{"delta": {"content": "NONE"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 7}},
            "[DONE]",
        )
        result = complete(FakeOpener(stream), "http://localhost", {}, {"stream": True})
        self.assertEqual(check({"exact": "NONE"}, result), "exact_pass")
        self.assertEqual(result["content"], "NONE")
        self.assertEqual(result["usage"], {"completion_tokens": 7})
        self.assertLessEqual(result["first_reasoning_s"], result["first_content_s"])

    def test_done_is_required_even_when_stop_arrives(self):
        stream = sse({"choices": [{"delta": {"content": "NONE"}, "finish_reason": "stop"}]})
        result = complete(FakeOpener(stream), "http://localhost", {}, {"stream": True})
        self.assertEqual(check({"exact": "NONE"}, result), "incomplete")

    def test_length_takes_precedence_over_correct_text(self):
        response = {"choices": [{"message": {"content": "NONE"}, "finish_reason": "length"}]}
        result = complete(FakeOpener(json.dumps(response).encode()), "http://localhost", {}, {"stream": False})
        self.assertEqual(check({"exact": "NONE"}, result), "length")
        self.assertNotIn("first_content_s", result)

    def test_http_error_stays_separate_from_content_failure(self):
        error = HTTPError("http://localhost", 503, "busy", {"Retry-After": "1"}, io.BytesIO())
        result = complete(FakeOpener(error), "http://localhost", {}, {"stream": True})
        self.assertEqual(check({}, result), "transport")
        self.assertEqual(result["retry_after"], "1")

    def test_required_forbidden_and_human_review(self):
        case = {"required": ["Mai", "thứ Hai"], "forbidden": ["thứ Sáu"]}
        result = {"content": "Mai, thứ Hai.", "finish_reason": "stop", "done": True}
        self.assertEqual(check(case, result), "needs_review")
        self.assertEqual(check(case, dict(result, content="Mai, thứ Sáu.")), "content")
        self.assertEqual(check(case, dict(result, content="Mai.")), "content")


if __name__ == "__main__":
    unittest.main()
