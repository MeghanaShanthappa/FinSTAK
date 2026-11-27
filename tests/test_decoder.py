"""Optional runtime tests for the decoder's GPU-independent masking math."""

from types import SimpleNamespace
import unittest

from stak_graph.extraction.backends import BackendError, SchemaLogitsProcessor

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is optional; run with the local inference environment")
class DecoderTests(unittest.TestCase):
    def processor(self):
        class Matcher:
            def __init__(self, grammar):
                self.accepted = []

            def accept_token(self, token):
                self.accepted.append(token)
                return token in (1, 31, 32, 63)

            def fill_next_token_bitmask(self, mask, row):
                mask[row, 0] = -2147483646
                mask[row, 1] = -2147483647

        grammar_module = SimpleNamespace(
            GrammarMatcher=Matcher,
            allocate_token_bitmask=lambda batch, vocab: torch.zeros((batch, (vocab + 31) // 32), dtype=torch.int32),
        )
        grammar = SimpleNamespace(tokenizer_info=SimpleNamespace(vocab_size=64))
        return SchemaLogitsProcessor(torch, grammar_module, grammar)

    def test_signed_bitmasks_preserve_allowed_tokens(self):
        result = self.processor()(torch.tensor([[50, 10]]), torch.zeros((1, 64)))
        self.assertEqual(torch.isfinite(result).nonzero()[:, 1].tolist(), [1, 31, 32, 63])

    def test_next_generated_token_is_consumed(self):
        processor = self.processor()
        processor(torch.tensor([[50, 10]]), torch.zeros((1, 64)))
        processor(torch.tensor([[50, 10, 31]]), torch.zeros((1, 64)))
        self.assertEqual(processor.matcher.accepted, [31])

    def test_invalid_token_is_an_error(self):
        processor = self.processor()
        processor(torch.tensor([[50, 10]]), torch.zeros((1, 64)))
        with self.assertRaises(BackendError):
            processor(torch.tensor([[50, 10, 2]]), torch.zeros((1, 64)))

    def test_batching_cannot_silently_corrupt_masks(self):
        with self.assertRaises(BackendError):
            self.processor()(torch.ones((2, 3), dtype=torch.long), torch.zeros((2, 64)))


if __name__ == "__main__":
    unittest.main()
