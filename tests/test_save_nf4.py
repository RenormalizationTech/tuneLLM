import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.save_nf4 import stream_hash, tensor_hash


class StreamingHashTests(unittest.TestCase):
    def test_file_stream_matches_sha256_across_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "checkpoint"
            content = b"packed NF4 state" * 100
            filename.write_bytes(content)
            self.assertEqual(stream_hash(filename, chunk_bytes=7), hashlib.sha256(content).hexdigest())

    def test_tensor_stream_is_chunk_independent_and_detects_bit_change(self):
        import torch
        original = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        self.assertEqual(tensor_hash(original, 7), tensor_hash(original, 100))
        self.assertEqual(tensor_hash(original, 7), hashlib.sha256(original.view(torch.uint8).numpy().tobytes()).hexdigest())
        changed = original.clone()
        changed[0, 0] = 1
        self.assertNotEqual(tensor_hash(original), tensor_hash(changed))


if __name__ == "__main__":
    unittest.main()
