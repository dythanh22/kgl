import unittest

from kgl.remote import MARKER, MarkerParser, last_marker, render
from kgl.state import SessionStore, redact_url, Session
from kgl.ws import encode_frame, parse_frame, OP_TEXT


class FrameTests(unittest.TestCase):
    def test_roundtrip_sizes(self):
        for n in (0, 5, 125, 126, 65535, 65536, 200000):
            payload = bytes(range(256)) * (n // 256) + bytes(n % 256)
            for mask in (True, False):
                buf = bytearray(encode_frame(OP_TEXT, payload, mask=mask))
                fin, op, data, used = parse_frame(buf)
                self.assertEqual((fin, op, data, used), (True, OP_TEXT, payload, len(buf)))

    def test_incomplete_returns_none(self):
        buf = bytearray(encode_frame(OP_TEXT, b"hello world"))
        for cut in range(len(buf)):
            self.assertIsNone(parse_frame(bytearray(buf[:cut])))


class MarkerTests(unittest.TestCase):
    def test_split_across_chunks_and_passthrough(self):
        p = MarkerParser()
        out = p.feed("hello\n@@KG")
        out += p.feed('L@@{"a": 1}\nprog 50%\r')
        out += p.feed("done\n")
        self.assertEqual(p.markers, [{"a": 1}])
        self.assertEqual(out, "hello\nprog 50%\rdone\n")

    def test_partial_line_not_marker_is_flushed(self):
        p = MarkerParser()
        self.assertEqual(p.feed("abc"), "abc")

    def test_bad_json_marker_passes_through(self):
        p = MarkerParser()
        self.assertEqual(p.feed(MARKER + "{bad\n"), MARKER + "{bad\n")

    def test_last_marker_and_render_quoting(self):
        code = render("x = __V__\n_emit({'x': x})\n", V="it's \"quoted\"\n")
        ns: dict = {}
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
        self.assertEqual(last_marker(buf.getvalue()), {"x": "it's \"quoted\"\n"})


class StateTests(unittest.TestCase):
    def test_redact(self):
        r = redact_url("https://h.example/proxy/abcdefghijklmnopqrstuv/x?token=SECRET")
        self.assertNotIn("SECRET", r)
        self.assertNotIn("abcdefghijklmnopqrstuv", r)

    def test_store_permissions(self):
        import os, stat, tempfile, pathlib
        d = pathlib.Path(tempfile.mkdtemp()) / "h"
        s = SessionStore(d)
        s.save(Session(base_url="http://x", kernel_id="k"))
        self.assertEqual(stat.S_IMODE(os.stat(s.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(d).st_mode), 0o700)
        self.assertEqual(s.load().kernel_id, "k")
        s.clear()
        self.assertIsNone(s.load())


if __name__ == "__main__":
    unittest.main()
