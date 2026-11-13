import unittest
from uuid import UUID

from alfa_search import SearchConfig
from alfa_search.baseline.indexing import prepare_index
from alfa_search.models import Material
from alfa_search.baseline.preprocessing import Preprocessor
from alfa_search.text import clean_text


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prep = Preprocessor()

    def test_lemmas_and_frequencies(self) -> None:
        self.assertEqual(self.prep.tokens("НАЛОГИ налогов налог"), ["налог"] * 3)
        self.assertEqual(self.prep.tokens("расчёт"), self.prep.tokens("расчет"))

    def test_business_terms_and_numbers(self) -> None:
        self.assertEqual(self.prep.tokens("1С НДС B2B CRM SEO ИИ ИП 2026"),
                         ["1с", "ндс", "b2b", "crm", "seo", "ии", "ип", "2026"])

    def test_empty_html_and_stop_words(self) -> None:
        for text in (None, "", "   ", "!!!", "и для the"):
            self.assertEqual(self.prep.tokens(text), [])
        self.assertEqual(clean_text("<p>Один</p><p>два&nbsp;три</p><script>bad</script>"), "Один два три")
        self.assertEqual(self.prep.tokens("мар<b>кетинг</b>"), ["маркетинг"])

    def test_field_counts_and_document_frequency(self) -> None:
        prepared = prepare_index([Material(UUID(int=1), "course", "Налоги налоги", tags=("налог",)),
                                  Material(UUID(int=1), "article", "Налоги")], self.prep)
        self.assertEqual(prepared.materials[0][2], [2, 0, 0, 1, 0])
        self.assertEqual(prepared.frequencies["налог"], 2)
        self.assertIn(("налог", "course", UUID(int=1), 1, 2), prepared.postings)
        self.assertEqual(prepared.averages, [1.5, 0, 0, 0.5, 0])
        with self.assertRaises(ValueError):
            prepare_index([Material(UUID(int=1), "course")] * 2, self.prep)

    def test_configuration_validation(self) -> None:
        for kwargs in ({"weights": (1,)}, {"b": (2,) * 5}, {"k1": 0},
                       {"k1": float("nan")}, {"weights": (float("inf"),) * 5}):
            with self.assertRaises(ValueError):
                SearchConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
