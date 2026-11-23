"""Tests voor de spraakannotatie (zonder spraakmodel, audio of GPU).

Test de stappen na de spraakherkenning: commando's vinden in een woordenlijst,
daar acties van maken (commando ... klaar, fout) en Vosk + Whisper combineren.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_asr_test as asr  # noqa: E402


def woorden(tekst, stap=0.5, begin=0.0):
    """Maak een nep-woordenlijst zoals een spraakmodel die teruggeeft."""
    return [{"start": round(begin + i * stap, 2), "end": round(begin + i * stap + 0.4, 2), "word": w}
            for i, w in enumerate(tekst.split())]


class TestFindCommands(unittest.TestCase):
    def test_exacte_commandos(self):
        cmds = asr.find_commands(woorden("start losdraaien klaar draai om klaar"))
        self.assertEqual([c["command"] for c in cmds], ["start losdraaien", "klaar", "draai om", "klaar"])

    def test_gesplitst_woord_van_vosk(self):
        cmds = asr.find_commands(woorden("start los draaien klaar"))
        self.assertEqual(cmds[0]["command"], "start losdraaien")

    def test_kleine_verstaanfout(self):
        cmds = asr.find_commands(woorden("start vrikken klaar"))
        self.assertEqual(cmds[0]["command"], "start wrikken")

    def test_gewone_tekst_geeft_geen_commando(self):
        cmds = asr.find_commands(woorden("het glijvoetje zit vast met dubbelzijdig tape"))
        self.assertEqual(cmds, [])


class TestBuildActions(unittest.TestCase):
    def test_commando_en_klaar_geven_een_actie(self):
        cmds = asr.find_commands(woorden("start losdraaien klaar", stap=1.0, begin=10.0))
        acties = asr.build_actions(cmds, "test.wav", fps=30)
        self.assertEqual(len(acties), 1)
        a = acties[0]
        self.assertEqual((a["label"], a["edim"]), ("screw_unscrew", "Disconnect"))
        self.assertEqual(a["start_frame"], 300)          # 10,0 s x 30 fps
        self.assertEqual(a["end_frame"], round(12.4 * 30))  # eind van 'klaar'

    def test_fout_schrapt_vorige_actie(self):
        cmds = asr.find_commands(woorden("draai om klaar vasthouden klaar fout"))
        acties = asr.build_actions(cmds, "test.wav", fps=30)
        self.assertEqual([a["label"] for a in acties], ["rotate_object"])

    def test_actie_zonder_klaar_wordt_gemarkeerd(self):
        cmds = asr.find_commands(woorden("draai om vasthouden klaar"))
        acties = asr.build_actions(cmds, "test.wav", fps=30)
        self.assertEqual(len(acties), 2)
        self.assertIn("klaar", acties[0]["notes"])


class TestCombine(unittest.TestCase):
    def test_vals_commando_van_vosk_wordt_verworpen(self):
        vosk = asr.find_commands(woorden("pak spatel klaar fout", begin=10.0))
        whisper_w = woorden("pak spatel klaar het is fijn", begin=10.2)
        whisper_c = asr.find_commands(whisper_w)
        acties, kept, dropped = asr.combine(vosk, whisper_c, whisper_w, "test.wav", fps=30)
        self.assertEqual([d["command"] for d in dropped], ["fout"])
        self.assertEqual(len(acties), 1)
        self.assertEqual(acties[0]["tool"], "spatel")

    def test_tijden_komen_van_vosk(self):
        vosk = asr.find_commands(woorden("draai om klaar", begin=20.0))
        whisper_w = woorden("draai om klaar", begin=19.2)  # Whisper iets te vroeg
        acties, _, _ = asr.combine(vosk, asr.find_commands(whisper_w), whisper_w, "test.wav", fps=30)
        self.assertEqual(acties[0]["start_s"], 20.0)

    def test_gereedschap_blijft_tot_leg_neer(self):
        tekst = "pak schroevendraaier kruiskop ph0 klaar start losdraaien klaar leg neer klaar draai om klaar"
        w = woorden(tekst)
        acties, _, _ = asr.combine(asr.find_commands(w), asr.find_commands(w), w, "test.wav", fps=30)
        tools = {a["label"]: a["tool"] for a in acties}
        self.assertIn("schroevendraaier", tools["screw_unscrew"])
        self.assertIn("PH0", tools["screw_unscrew"])
        self.assertEqual(tools["rotate_object"], "")


class TestExcel(unittest.TestCase):
    def test_excel_bestand_wordt_gemaakt(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl niet geinstalleerd (alleen in .venv-asr)")
        import csv
        import asr_naar_excel
        w = woorden("draai om klaar")
        acties = asr.build_actions(asr.find_commands(w), "test.wav", fps=30)
        with tempfile.TemporaryDirectory() as tmp:
            map_ = Path(tmp) / "test"
            map_.mkdir()
            velden = ["video", "start_frame", "end_frame", "label", "edim", "start_s", "end_s", "notes"]
            with open(map_ / "vosk_actions.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=velden)
                wr.writeheader()
                wr.writerows(acties)
            sys.argv = ["asr_naar_excel.py", str(map_)]
            asr_naar_excel.main()
            self.assertTrue((map_ / "test_resultaten.xlsx").exists())


if __name__ == "__main__":
    unittest.main()
