import unittest
from types import SimpleNamespace
from eval.codex.audio_emotion_analyzer import parse_emotion_response


class ResponseValidation(unittest.TestCase):
    def response(self, text, reason='STOP'):
        return SimpleNamespace(text=text,candidates=[SimpleNamespace(finish_reason=reason)])

    def test_truncation_is_failure_not_label(self):
        with self.assertRaisesRegex(ValueError, 'MAX_TOKENS'):
            parse_emotion_response(self.response('{}','MAX_TOKENS'),'clip.wav')

    def test_empty_is_failure(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            parse_emotion_response(self.response(''),'clip.wav')

    def test_missing_candidate_is_failure(self):
        with self.assertRaises(ValueError):
            parse_emotion_response(SimpleNamespace(text='{}',candidates=[]),'clip.wav')

    def test_valid_json_uses_actual_filename(self):
        r=parse_emotion_response(self.response('{"filename":"invented.wav","primary_emotion":"uncertain","alternative_emotion":null,"confidence":20,"audible_evidence":"The tone is ambiguous."}'),'clip.wav')
        self.assertEqual(r.filename,'clip.wav')
        self.assertEqual(r.primary_emotion,'uncertain')

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            parse_emotion_response(self.response('{"primary_emotion":"madeup","alternative_emotion":null,"confidence":20,"audible_evidence":"x"}'),'clip.wav')


if __name__=='__main__':unittest.main()
