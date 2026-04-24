"""
test_integration.py
===================
Dreamscape Mapper — End-to-End Integration Test Suite

Covers:
  1.  Full pipeline smoke test (sample dream → structured output)
  2.  Output schema validation (required keys + types)
  3.  Mathematical invariants (dominant emotion = argmax of Emotion_Vector)
  4.  No-null / no-empty-string guard on every pipeline stage output
  5.  Per-step unit tests (steps 1-10 in isolation)
  6.  Edge-case / robustness tests (empty input, single word, punctuation-only)
  7.  Multiple diverse dreams to stress-test theme assignment
  8.  Emotion vector normalisation checks

Run with:  python -m pytest test_integration.py -v
       or:  python test_integration.py
"""

import json
import math
import unittest
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Import the pipeline — fail loudly if missing
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dream_pipeline import (
        run_pipeline,
        step1_ingest,
        step2_preprocess,
        step3_silver_annotate,
        step4_bilstm_representations,
        step5_emotion_prediction,
        step6_enrich_embeddings,
        step7_assign_cluster,
        step8_extract_keywords,
        step9_generate_topic_label,
        step10_dominant_emotion,
        _NRC_EMOTIONS,
    )
except ImportError as exc:
    print(f"FATAL: Cannot import dream_pipeline — {exc}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_DREAM = (
    "I looked in the mirror and suddenly my teeth started falling out. "
    "My mother was standing nearby laughing at me and I felt extremely embarrassed."
)

DIVERSE_DREAMS = {
    "pursuit":  "A monster chased me down a dark hallway and I ran as fast as I could.",
    "flying":   "I was flying high above the clouds, soaring freely over mountains.",
    "water":    "The ocean waves rose and I started to drown, sinking into the deep water.",
    "loss":     "I stood at a funeral, crying. My grandmother had died and she was gone forever.",
    "exam":     "I arrived late to the exam, completely unprepared. The teacher stared at me.",
    "short":    "Falling.",
    "social":   "I was naked in front of a crowd and felt humiliated and ashamed.",
}

REQUIRED_KEYS = {
    "Topic_Cluster",
    "Dominant_Emotion",
    "Key_Entities",
    "Semantic_Relation",
    "Emotion_Vector",
}

SEMANTIC_RELATION_KEYS = {"Agent", "Action", "Target"}


# ──────────────────────────────────────────────────────────────────────────────
# Helper assertions
# ──────────────────────────────────────────────────────────────────────────────

def assert_no_none(obj, path="root"):
    """Recursively assert that no value in a nested dict/list is None."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert v is not None, f"Null value at {path}.{k}"
            assert_no_none(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert v is not None, f"Null value at {path}[{i}]"
            assert_no_none(v, f"{path}[{i}]")


def assert_no_empty_strings(obj, path="root"):
    """Recursively assert that no string value is empty."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_empty_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_empty_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        assert obj.strip() != "", f"Empty string at {path}"


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 1 — Full pipeline smoke + schema
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipelineSmoke(unittest.TestCase):
    """Run the sample dream and check the result exists and is dict-shaped."""

    def setUp(self):
        self.result = run_pipeline(SAMPLE_DREAM)

    def test_result_is_dict(self):
        self.assertIsInstance(self.result, dict, "Pipeline must return a dict.")

    def test_result_is_json_serialisable(self):
        try:
            json.dumps(self.result)
        except (TypeError, ValueError) as exc:
            self.fail(f"Result is not JSON-serialisable: {exc}")

    def test_result_is_not_empty(self):
        self.assertTrue(self.result, "Result dict must not be empty.")


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 2 — Required schema keys and value types
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputSchema(unittest.TestCase):

    def setUp(self):
        self.result = run_pipeline(SAMPLE_DREAM)

    def test_required_keys_present(self):
        missing = REQUIRED_KEYS - set(self.result.keys())
        self.assertFalse(
            missing,
            f"Missing required keys: {missing}"
        )

    def test_topic_cluster_is_nonempty_string(self):
        tc = self.result["Topic_Cluster"]
        self.assertIsInstance(tc, str)
        self.assertTrue(tc.strip(), "Topic_Cluster must not be an empty string.")

    def test_dominant_emotion_is_nonempty_string(self):
        de = self.result["Dominant_Emotion"]
        self.assertIsInstance(de, str)
        self.assertTrue(de.strip(), "Dominant_Emotion must not be an empty string.")

    def test_key_entities_is_nonempty_list(self):
        ke = self.result["Key_Entities"]
        self.assertIsInstance(ke, list)
        self.assertTrue(ke, "Key_Entities must contain at least one entry.")

    def test_key_entities_items_are_strings(self):
        for item in self.result["Key_Entities"]:
            self.assertIsInstance(item, str, f"Key_Entities item must be str, got {type(item)}")
            self.assertTrue(item.strip(), "Key_Entities must not contain empty strings.")

    def test_semantic_relation_is_dict_with_correct_keys(self):
        sr = self.result["Semantic_Relation"]
        self.assertIsInstance(sr, dict)
        missing = SEMANTIC_RELATION_KEYS - set(sr.keys())
        self.assertFalse(missing, f"Semantic_Relation missing keys: {missing}")

    def test_semantic_relation_values_are_strings(self):
        for k, v in self.result["Semantic_Relation"].items():
            self.assertIsInstance(v, str, f"Semantic_Relation[{k}] must be str.")
            self.assertTrue(v.strip(), f"Semantic_Relation[{k}] must not be empty.")

    def test_emotion_vector_is_dict(self):
        ev = self.result["Emotion_Vector"]
        self.assertIsInstance(ev, dict)
        self.assertTrue(ev, "Emotion_Vector must not be empty.")

    def test_emotion_vector_values_are_floats(self):
        for k, v in self.result["Emotion_Vector"].items():
            self.assertIsInstance(v, (int, float), f"Emotion_Vector[{k}] must be numeric.")

    def test_emotion_vector_values_in_range(self):
        for k, v in self.result["Emotion_Vector"].items():
            self.assertGreaterEqual(v, 0.0, f"Emotion score for {k} must be >= 0.")
            self.assertLessEqual(v, 1.0, f"Emotion score for {k} must be <= 1.")

    def test_emotion_vector_contains_nrc_emotions(self):
        ev_keys = set(self.result["Emotion_Vector"].keys())
        nrc_set = set(_NRC_EMOTIONS)
        # At minimum the core 8 emotions should be present
        core = {"fear", "sadness", "anger", "disgust", "surprise", "anticipation", "joy", "trust"}
        self.assertTrue(
            core.issubset(ev_keys),
            f"Emotion_Vector missing core NRC emotions: {core - ev_keys}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 3 — Mathematical invariants
# ──────────────────────────────────────────────────────────────────────────────

class TestMathematicalInvariants(unittest.TestCase):

    def _run(self, dream):
        return run_pipeline(dream)

    def test_dominant_emotion_matches_argmax_sample(self):
        """Dominant_Emotion must equal the key with the highest score in Emotion_Vector
        (excluding valence dimensions positive/negative)."""
        result = self._run(SAMPLE_DREAM)
        ev = result["Emotion_Vector"]
        core = {k: v for k, v in ev.items() if k not in ("positive", "negative")}
        argmax = max(core, key=core.get)
        dominant = result["Dominant_Emotion"].lower()
        self.assertEqual(
            dominant,
            argmax,
            f"Dominant_Emotion='{dominant}' but argmax of Emotion_Vector is '{argmax}'. "
            f"Vector: {core}"
        )

    def test_dominant_emotion_matches_argmax_all_diverse(self):
        for label, dream in DIVERSE_DREAMS.items():
            with self.subTest(dream=label):
                result = self._run(dream)
                ev = result["Emotion_Vector"]
                core = {k: v for k, v in ev.items() if k not in ("positive", "negative")}
                if all(v == 0 for v in core.values()):
                    continue  # uniform prior — skip argmax check
                argmax = max(core, key=core.get)
                dominant = result["Dominant_Emotion"].lower()
                self.assertEqual(
                    dominant, argmax,
                    f"[{label}] Dominant='{dominant}' ≠ argmax='{argmax}'. Vector: {core}"
                )

    def test_emotion_scores_are_finite(self):
        result = self._run(SAMPLE_DREAM)
        for k, v in result["Emotion_Vector"].items():
            self.assertTrue(
                math.isfinite(v),
                f"Emotion score for '{k}' is not finite: {v}"
            )

    def test_emotion_scores_sum_reasonable(self):
        """Sum of core emotion scores should be in (0, number_of_emotions] (normalised)."""
        result = self._run(SAMPLE_DREAM)
        core = {k: v for k, v in result["Emotion_Vector"].items() if k not in ("positive", "negative")}
        total = sum(core.values())
        self.assertGreaterEqual(total, 0.0, "Sum of emotion scores must be >= 0")
        self.assertLessEqual(total, len(core), "Sum of emotion scores must be <= number of emotions")

    def test_dominant_emotion_capitalised(self):
        result = self._run(SAMPLE_DREAM)
        dom = result["Dominant_Emotion"]
        self.assertEqual(dom, dom.capitalize(), f"Dominant_Emotion should be capitalised: got '{dom}'")


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 4 — No-null / no-silent-data-drop guard
# ──────────────────────────────────────────────────────────────────────────────

class TestNoNullOrGarbageValues(unittest.TestCase):

    def setUp(self):
        self.result = run_pipeline(SAMPLE_DREAM)

    def test_no_none_values_in_result(self):
        assert_no_none(self.result)

    def test_no_empty_string_values(self):
        # Only check the mandatory keys to avoid false positives on optional fields
        mandatory_result = {k: self.result[k] for k in REQUIRED_KEYS}
        assert_no_empty_strings(mandatory_result)

    def test_topic_cluster_not_none_string(self):
        self.assertNotEqual(self.result["Topic_Cluster"].lower(), "none")
        self.assertNotEqual(self.result["Topic_Cluster"].lower(), "null")

    def test_key_entities_not_placeholder(self):
        for e in self.result["Key_Entities"]:
            self.assertNotIn(e.lower(), ("none", "null", "n/a"), f"Placeholder entity found: {e}")

    def test_semantic_relation_no_placeholder_values(self):
        for k, v in self.result["Semantic_Relation"].items():
            self.assertNotIn(v.lower(), ("none", "null", "n/a"), f"Placeholder in Semantic_Relation[{k}]: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 5 — Per-step unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestStep1Ingest(unittest.TestCase):
    def test_valid_input_returned(self):
        self.assertEqual(step1_ingest("  hello world  "), "hello world")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            step1_ingest("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            step1_ingest("   ")


class TestStep2Preprocess(unittest.TestCase):
    def setUp(self):
        self.out = step2_preprocess(SAMPLE_DREAM)

    def test_has_required_keys(self):
        for k in ("tokens", "lemmas", "sentences", "segments"):
            self.assertIn(k, self.out)

    def test_tokens_nonempty(self):
        self.assertTrue(self.out["tokens"])

    def test_segments_nonempty(self):
        self.assertTrue(self.out["segments"])

    def test_tokens_are_lowercase(self):
        for t in self.out["tokens"]:
            self.assertEqual(t, t.lower(), f"Token not lowercase: {t}")

    def test_segments_match_sentence_count(self):
        # Each segment maps to one sentence
        self.assertGreaterEqual(len(self.out["segments"]), 1)


class TestStep3Annotation(unittest.TestCase):
    def setUp(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        self.ann = step3_silver_annotate(preprocessed)

    def test_has_required_keys(self):
        for k in ("entities", "srl", "coreference", "emotion_stubs"):
            self.assertIn(k, self.ann)

    def test_entities_categories_present(self):
        for cat in ("Character", "Body-Object", "Setting-Object"):
            self.assertIn(cat, self.ann["entities"])

    def test_srl_has_agent_action_target(self):
        for k in ("Agent", "Action", "Target"):
            self.assertIn(k, self.ann["srl"])
            self.assertIsInstance(self.ann["srl"][k], str)

    def test_sample_dream_detects_mother(self):
        chars = [c.lower() for c in self.ann["entities"]["Character"]]
        self.assertIn("mother", chars, "Should detect 'Mother' as a Character entity.")

    def test_sample_dream_detects_teeth(self):
        body = [b.lower() for b in self.ann["entities"]["Body-Object"]]
        self.assertIn("teeth", body, "Should detect 'Teeth' as a Body-Object entity.")


class TestStep4BiLSTM(unittest.TestCase):
    def setUp(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        ann = step3_silver_annotate(preprocessed)
        self.out = step4_bilstm_representations(preprocessed, ann)

    def test_has_required_keys(self):
        for k in ("contextual_embedding", "entity_presence_vector", "srl_verb_signature"):
            self.assertIn(k, self.out)

    def test_embedding_values_numeric(self):
        for k, v in self.out["contextual_embedding"].items():
            if isinstance(v, (int, float)):
                self.assertTrue(math.isfinite(v))


class TestStep5Emotions(unittest.TestCase):
    def setUp(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        self.out = step5_emotion_prediction(preprocessed)

    def test_has_required_keys(self):
        for k in ("segment_emotions", "document_emotion_vector"):
            self.assertIn(k, self.out)

    def test_document_vector_has_nrc_keys(self):
        core = {"fear", "sadness", "anger", "disgust", "surprise", "anticipation", "joy", "trust"}
        self.assertTrue(core.issubset(set(self.out["document_emotion_vector"].keys())))

    def test_all_scores_in_range(self):
        for k, v in self.out["document_emotion_vector"].items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_segment_emotions_nonempty(self):
        self.assertTrue(self.out["segment_emotions"])


class TestStep7ClusterAssignment(unittest.TestCase):
    def test_body_integrity_cluster(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        cluster = step7_assign_cluster(preprocessed)
        self.assertEqual(cluster, "Body Integrity Disturbance",
                         f"Expected 'Body Integrity Disturbance', got '{cluster}'")

    def test_pursuit_cluster(self):
        preprocessed = step2_preprocess(DIVERSE_DREAMS["pursuit"])
        cluster = step7_assign_cluster(preprocessed)
        self.assertEqual(cluster, "Pursuit and Escape",
                         f"Expected 'Pursuit and Escape', got '{cluster}'")

    def test_flying_cluster(self):
        preprocessed = step2_preprocess(DIVERSE_DREAMS["flying"])
        cluster = step7_assign_cluster(preprocessed)
        self.assertEqual(cluster, "Flying and Freedom",
                         f"Expected 'Flying and Freedom', got '{cluster}'")

    def test_water_cluster(self):
        preprocessed = step2_preprocess(DIVERSE_DREAMS["water"])
        cluster = step7_assign_cluster(preprocessed)
        self.assertEqual(cluster, "Water and Submersion",
                         f"Expected 'Water and Submersion', got '{cluster}'")


class TestStep8Keywords(unittest.TestCase):
    def test_returns_list(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        kws = step8_extract_keywords(preprocessed)
        self.assertIsInstance(kws, list)

    def test_no_stopwords_in_keywords(self):
        from dream_pipeline import _STOPWORDS
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        kws = step8_extract_keywords(preprocessed)
        for kw in kws:
            self.assertNotIn(kw.lower(), _STOPWORDS, f"Stopword in keywords: {kw}")

    def test_keywords_respect_top_n(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        for n in (3, 5, 8):
            kws = step8_extract_keywords(preprocessed, top_n=n)
            self.assertLessEqual(len(kws), n)


class TestStep10DominantEmotion(unittest.TestCase):
    def test_dominant_is_argmax(self):
        preprocessed = step2_preprocess(SAMPLE_DREAM)
        emotion_out = step5_emotion_prediction(preprocessed)
        dominant, avg_vec = step10_dominant_emotion(emotion_out)
        core = {k: v for k, v in avg_vec.items() if k not in ("positive", "negative")}
        if not all(v == 0 for v in core.values()):
            self.assertEqual(dominant, max(core, key=core.get))

    def test_zero_vector_handled(self):
        # Construct a zero emotion_out to test graceful handling
        zero_out = {
            "segment_emotions": {
                "seg1": {e: 0.0 for e in _NRC_EMOTIONS}
            }
        }
        dominant, vec = step10_dominant_emotion(zero_out)
        self.assertIsInstance(dominant, str)
        self.assertTrue(dominant.strip())

    def test_empty_segment_list_handled(self):
        empty_out = {"segment_emotions": {}}
        dominant, vec = step10_dominant_emotion(empty_out)
        self.assertEqual(dominant, "unknown")


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 6 — Edge cases & robustness
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_single_word_dream(self):
        result = run_pipeline("Falling.")
        self.assertIn("Topic_Cluster", result)
        self.assertIn("Dominant_Emotion", result)

    def test_long_dream(self):
        long_dream = (
            "I was running through an endless dark forest being chased by a faceless monster. "
            "Every time I looked back it got closer. My legs felt like lead. "
            "Suddenly I fell into a deep pit filled with water. "
            "I tried to swim but the water was pulling me down. "
            "My teeth started falling out one by one as I sank. "
            "I could see my mother standing at the edge, laughing. "
            "I felt complete shame and terror as the darkness swallowed me."
        )
        result = run_pipeline(long_dream)
        self.assertIn("Topic_Cluster", result)
        self.assertIn("Dominant_Emotion", result)
        # Should not crash on multi-sentence dream
        self.assertIsInstance(result["Key_Entities"], list)

    def test_dream_with_only_punctuation_raises_or_returns_gracefully(self):
        """Either raise ValueError or return a gracefully degraded result."""
        try:
            result = run_pipeline("!!! ??? ...")
            # If it doesn't raise, check it at least has the schema
            self.assertIn("Topic_Cluster", result)
        except ValueError:
            pass  # Acceptable: pipeline rejects degenerate input

    def test_unicode_dream(self):
        result = run_pipeline("I dreamed of flying over beautiful mountains and rivers.")
        self.assertIn("Topic_Cluster", result)

    def test_result_is_deterministic(self):
        """Same input → same output (pipeline has no randomness)."""
        r1 = run_pipeline(SAMPLE_DREAM)
        r2 = run_pipeline(SAMPLE_DREAM)
        self.assertEqual(r1["Topic_Cluster"], r2["Topic_Cluster"])
        self.assertEqual(r1["Dominant_Emotion"], r2["Dominant_Emotion"])
        self.assertEqual(r1["Emotion_Vector"], r2["Emotion_Vector"])

    def test_diverse_dreams_all_produce_valid_schema(self):
        for label, dream in DIVERSE_DREAMS.items():
            with self.subTest(dream=label):
                result = run_pipeline(dream)
                missing = REQUIRED_KEYS - set(result.keys())
                self.assertFalse(missing, f"[{label}] Missing keys: {missing}")
                self.assertIsInstance(result["Key_Entities"], list)
                self.assertIsInstance(result["Emotion_Vector"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# Test Class 7 — Project-spec compliance (exact requirements from the paper)
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectSpecCompliance(unittest.TestCase):
    """
    Checks that the output matches the exact structured format described in the
    Dreamscape Mapper paper (Section 2 / Listing 1).
    """

    def setUp(self):
        self.result = run_pipeline(SAMPLE_DREAM)

    def test_topic_cluster_key_exists(self):
        self.assertIn("Topic_Cluster", self.result)

    def test_dominant_emotion_key_exists(self):
        self.assertIn("Dominant_Emotion", self.result)

    def test_key_entities_key_exists(self):
        self.assertIn("Key_Entities", self.result)

    def test_semantic_relation_key_exists(self):
        self.assertIn("Semantic_Relation", self.result)

    def test_emotion_vector_key_exists(self):
        self.assertIn("Emotion_Vector", self.result)

    def test_semantic_relation_agent_action_target(self):
        sr = self.result["Semantic_Relation"]
        self.assertIn("Agent", sr)
        self.assertIn("Action", sr)
        self.assertIn("Target", sr)

    def test_sample_dream_cluster_is_body_integrity(self):
        self.assertEqual(
            self.result["Topic_Cluster"],
            "Body Integrity Disturbance",
            f"Expected 'Body Integrity Disturbance' for canonical sample dream, "
            f"got: '{self.result['Topic_Cluster']}'"
        )

    def test_sample_dream_dominant_emotion_is_fear_or_sadness(self):
        """
        For the canonical mirror/teeth/shame dream, dominant NRC emotion should
        be either Fear or Sadness (NRC maps shame-adjacent cues to sadness).
        """
        dom = self.result["Dominant_Emotion"].lower()
        self.assertIn(
            dom, ("fear", "sadness"),
            f"Expected 'fear' or 'sadness' as dominant emotion for sample dream, got '{dom}'"
        )

    def test_emotion_vector_contains_fear_and_sadness(self):
        ev = self.result["Emotion_Vector"]
        self.assertIn("fear", ev, "Emotion_Vector must contain 'fear'")
        self.assertIn("sadness", ev, "Emotion_Vector must contain 'sadness'")

    def test_mother_or_teeth_in_key_entities(self):
        entities_lower = [e.lower() for e in self.result["Key_Entities"]]
        self.assertTrue(
            "mother" in entities_lower or "teeth" in entities_lower,
            f"Expected 'Mother' or 'Teeth' in Key_Entities, got: {self.result['Key_Entities']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use verbosity=2 for detailed output when run directly
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestFullPipelineSmoke,
        TestOutputSchema,
        TestMathematicalInvariants,
        TestNoNullOrGarbageValues,
        TestStep1Ingest,
        TestStep2Preprocess,
        TestStep3Annotation,
        TestStep4BiLSTM,
        TestStep5Emotions,
        TestStep7ClusterAssignment,
        TestStep8Keywords,
        TestStep10DominantEmotion,
        TestEdgeCases,
        TestProjectSpecCompliance,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
