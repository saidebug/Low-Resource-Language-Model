import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import check_output, extract_numbers, script_ratio  # noqa: E402


class CheckTests(unittest.TestCase):
    def test_numbers_from_other_scripts(self):
        self.assertEqual(extract_numbers("௩௦ நவம்பர்"), ["30"])
        self.assertEqual(extract_numbers("३० नवंबर"), ["30"])
        self.assertEqual(extract_numbers("Rs 1,500"), ["1500"])

    def test_script_ratio(self):
        self.assertGreater(script_ratio("बिजली का बिल", "hindi"), 0.95)
        self.assertLess(script_ratio("electricity bill", "hindi"), 0.1)
        self.assertGreater(script_ratio("மின் கட்டணம்", "tamil"), 0.95)

    def test_good_answer_passes(self):
        src = "आवेदन 30 नवंबर तक करें और अपना बैंक खाता साथ लाएं।"
        out = "30 नवंबर तक आवेदन दें और बैंक खाता लाएं।"
        self.assertTrue(check_output(src, out, "hindi")["ok"])

    def test_missing_number_fails(self):
        src = "आवेदन 30 नवंबर तक करें और अपना बैंक खाता साथ लाएं।"
        res = check_output(src, "नवंबर तक आवेदन दें और बैंक खाता लाएं।", "hindi")
        self.assertFalse(res["ok"])
        self.assertTrue(any("missing" in p for p in res["problems"]))

    def test_invented_number_fails(self):
        src = "दवा दिन में दो बार लें और खाने के बाद पानी पिएं।"
        res = check_output(src, "दवा दिन में 3 बार लें और खाने के बाद पानी पिएं।", "hindi")
        self.assertFalse(res["ok"])

    def test_wrong_script_fails(self):
        src = "बिजली का बिल समय पर भरें वरना जुर्माना लगेगा।"
        res = check_output(src, "Pay your electricity bill on time or you will be fined.", "hindi")
        self.assertFalse(res["ok"])


if __name__ == "__main__":
    unittest.main()
