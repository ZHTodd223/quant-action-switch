import unittest
from pathlib import Path


class GenerateInterpolatedResponsesTest(unittest.TestCase):
    def test_mechanism_generator_is_in_memory_and_bounded(self):
        project = Path(__file__).resolve().parents[1]
        text = (project / "scripts/generate_interpolated_responses.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if not 0.0 <= args.alpha <= 1.0', text)
        self.assertIn('current.lerp(source, args.alpha)', text)
        self.assertIn('model.layers.{args.layer}.mlp.{matrix}.weight', text)
        self.assertIn('local_files_only=True', text)
        self.assertNotIn('save_pretrained', text)


if __name__ == "__main__":
    unittest.main()
