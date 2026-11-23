"""Spraakannotatie: test spraakmodellen (Vosk + faster-whisper) op een eigen opname.

Gebruik (vanuit de projectmap):
    run_asr_test.bat data\\opname_3min.m4a --combi small --vanaf 88
    run_asr_test.bat data\\opname_3min.m4a --models vosk,small,medium

Wat het doet, per model:
  1. transcribeert de audio met tijdstempels per woord
  2. zoekt de commando's uit het spreekprotocol
  3. maakt acties (commando ... klaar) met label en eDIM-categorie
Uitvoer in outputs/<opname>/:
  <model>_words.csv     elk herkend woord met begin- en eindtijd
  <model>_commands.csv  gevonden commando's
  <model>_actions.csv   acties in het formaat van MediaPipe_annotering.md (+ edim)
  summary.csv           rekentijd en aantallen per model
"""
import argparse
import csv
import difflib
import json
import re
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Spreekprotocol: commando -> (label, eDIM-categorie)
# ---------------------------------------------------------------------------
COMMANDS = {
    "zoek verbinding": ("identify", "Identity"),
    "draai om": ("rotate_object", "Manipulate"),
    "vasthouden": ("fixate", "Fix"),
    "pak schroevendraaier": ("tool_pick_up", "Tool Change"),
    "pak spatel": ("tool_pick_up", "Tool Change"),
    "pak tang": ("tool_pick_up", "Tool Change"),
    "pak sleutel": ("tool_pick_up", "Tool Change"),
    "pak hamer": ("tool_pick_up", "Tool Change"),
    "leg neer": ("tool_put_down", "Tool Change"),
    "plaats gereedschap": ("tool_position", "Tool Position"),
    "start losdraaien": ("screw_unscrew", "Disconnect"),
    "start vastdraaien": ("screw_tighten", "Disconnect"),
    "start wrikken": ("pry_component", "Disconnect"),
    "start loskoppelen": ("disconnect_other", "Disconnect"),
    "verwijder onderdeel": ("component_remove", "Removal"),
}
END_WORD = "klaar"
CANCEL_WORD = "fout"
ALL_PHRASES = list(COMMANDS) + [END_WORD, CANCEL_WORD]

# Hint voor Whisper: commando's + vakwoorden uit de documenten
HOTWORDS = ("zoek verbinding draai om vasthouden pak schroevendraaier spatel tang leg neer "
            "plaats gereedschap start losdraaien wrikken loskoppelen verwijder onderdeel klaar fout "
            "kruiskop torx inbus spudger borgveertang steeksleutel soldeerbout snap-fit")

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def load_audio(path):
    """Lees m4a/mp3/wav/mp4 in als 16 kHz mono float32 (via faster-whisper/PyAV)."""
    from faster_whisper import decode_audio
    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def to_pcm16(audio):
    import numpy as np
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Modellen
# ---------------------------------------------------------------------------
def run_vosk(audio, model_dir, use_grammar=True):
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    if not Path(model_dir).exists():
        sys.exit(f"Vosk-model niet gevonden: {model_dir}\n"
                 "Download vosk-model-small-nl-0.22 van https://alphacephei.com/vosk/models "
                 "en pak het uit in de map models/.")
    model = Model(str(model_dir))
    if use_grammar:
        vosk_phrases = [p.replace("losdraaien", "los draaien").replace("vastdraaien", "vast draaien")
                        for p in ALL_PHRASES]
        rec = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(vosk_phrases + ["[unk]"]))
    else:
        rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    pcm = to_pcm16(audio)
    words = []
    step = 8000  # 0,25 s per stuk
    for i in range(0, len(pcm), step):
        if rec.AcceptWaveform(pcm[i:i + step]):
            words += json.loads(rec.Result()).get("result", [])
    words += json.loads(rec.FinalResult()).get("result", [])
    return [{"start": w["start"], "end": w["end"], "word": w["word"], "conf": w.get("conf", "")}
            for w in words if w["word"] != "[unk]"]


def run_whisper(audio, size, use_hint=True):
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)
    kwargs = dict(language="nl", word_timestamps=True, vad_filter=True, beam_size=5)
    if use_hint:
        kwargs["hotwords"] = HOTWORDS
    segments, _ = model.transcribe(audio, **kwargs)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"start": round(w.start, 2), "end": round(w.end, 2),
                          "word": w.word.strip(), "conf": round(w.probability, 3)})
    return words


# ---------------------------------------------------------------------------
# Van woorden naar commando's en acties
# ---------------------------------------------------------------------------
def norm(text):
    return re.sub(r"[^a-z0-9à-ÿ\- ]", "", text.lower()).strip()


def find_commands(words, threshold=0.8):
    """Zoek commando's in de woordenlijst; vangt kleine schrijffouten op.

    Vergelijkt zonder spaties; bij commando's van 2+ woorden mag er 1 woord meer zijn,
    zodat 'start los draaien' (3 woorden) en 'start losdraaien' (2) allebei passen.
    """
    toks = [norm(w["word"]) for w in words]
    found, i = [], 0
    while i < len(words):
        best = None
        for phrase in ALL_PHRASES:
            target = phrase.replace(" ", "")
            n0 = len(phrase.split())
            for n in ((n0,) if n0 == 1 else (n0, n0 + 1)):
                if i + n > len(words):
                    continue
                cand = "".join(toks[i:i + n])
                score = difflib.SequenceMatcher(None, cand, target).ratio()
                if score >= threshold and (best is None or score > best[1]):
                    best = (phrase, score, n)
        if best:
            phrase, score, n = best
            found.append({"start": round(words[i]["start"], 2), "end": round(words[i + n - 1]["end"], 2),
                          "command": phrase, "heard": " ".join(w["word"] for w in words[i:i + n]),
                          "score": round(score, 2)})
            i += n
        else:
            i += 1
    return found


def build_actions(commands, video_name, fps):
    """Commando opent een actie, 'klaar' sluit hem, 'fout' schrapt de laatste."""
    actions, open_cmd = [], None
    for c in commands:
        if c["command"] == CANCEL_WORD:
            if open_cmd:
                open_cmd = None
            elif actions:
                actions.pop()
        elif c["command"] == END_WORD:
            if open_cmd:
                label, edim = COMMANDS[open_cmd["command"]]
                actions.append({"video": video_name,
                                "start_frame": round(open_cmd["start"] * fps),
                                "end_frame": round(c["end"] * fps),
                                "label": label, "edim": edim,
                                "start_s": open_cmd["start"], "end_s": c["end"],
                                "notes": ""})
                open_cmd = None
        else:
            if open_cmd:  # vorige actie zonder 'klaar' -> sluiten bij dit commando
                label, edim = COMMANDS[open_cmd["command"]]
                actions.append({"video": video_name,
                                "start_frame": round(open_cmd["start"] * fps),
                                "end_frame": round(c["start"] * fps),
                                "label": label, "edim": edim,
                                "start_s": open_cmd["start"], "end_s": c["start"],
                                "notes": "geen 'klaar' gehoord"})
            open_cmd = c
    if open_cmd:
        label, edim = COMMANDS[open_cmd["command"]]
        actions.append({"video": video_name, "start_frame": round(open_cmd["start"] * fps),
                        "end_frame": "", "label": label, "edim": edim,
                        "start_s": open_cmd["start"], "end_s": "",
                        "notes": "geen 'klaar' gehoord (einde opname)"})
    return actions


# ---------------------------------------------------------------------------
# Combinatie: Vosk (commando's + tijden) + Whisper (controle, tekst, gereedschap)
# ---------------------------------------------------------------------------
TOOL_WORDS = {
    "schroevendraaier": "schroevendraaier", "kruiskop": "kruiskop", "torx": "torx", "inbus": "inbus",
    "pozidriv": "pozidriv", "spatel": "spatel", "spudger": "spudger", "plectrum": "plectrum",
    "tang": "tang", "puntang": "puntang", "borgveertang": "borgveertang", "kniptang": "kniptang",
    "pincet": "pincet", "sleutel": "sleutel", "steeksleutel": "steeksleutel", "dopsleutel": "dopsleutel",
    "inbussleutel": "inbussleutel", "hamer": "hamer", "soldeerbout": "soldeerbout", "hand": "hand",
}
CODE_RE = re.compile(r"^(ph|pz|t|m)\d{1,2}$|^\d+(mm)?$")


def words_between(words, t0, t1):
    return [w for w in words if t0 <= (w["start"] + w["end"]) / 2 <= t1]


def tool_from_words(ws):
    """Zoek gereedschap + maat/code in een stuk tekst, bijv. 'kruiskop PH0'."""
    names, codes = [], []
    for w in ws:
        t = norm(w["word"]).replace(" ", "")
        if t in TOOL_WORDS and TOOL_WORDS[t] not in names:
            names.append(TOOL_WORDS[t])
        elif CODE_RE.match(t):
            codes.append(t.upper())
    return " ".join(names + codes)


def combine(vosk_cmds, whisper_cmds, whisper_words, video_name, fps, max_shift=3.0):
    """Houd alleen Vosk-commando's die Whisper ook rond die tijd hoorde; gebruik Vosk-tijden."""
    kept, dropped, used = [], [], set()
    for c in vosk_cmds:
        # dichtstbijzijnde nog niet gebruikte Whisper-match met hetzelfde commando
        best = None
        for j, w in enumerate(whisper_cmds):
            if j in used or w["command"] != c["command"]:
                continue
            d = abs(w["start"] - c["start"])
            if d <= max_shift and (best is None or d < best[1]):
                best = (j, d)
        if best:
            used.add(best[0])
            kept.append(c)
        else:
            dropped.append(c)

    actions = build_actions(kept, video_name, fps)
    tool_in_hand = ""
    for a in actions:
        t0 = a["start_s"]
        t1 = a["end_s"] if a["end_s"] != "" else t0 + 10
        ws = words_between(whisper_words, t0 - 1.0, t1)  # Whisper zet woorden vaak iets te vroeg
        tekst = " ".join(w["word"] for w in ws)
        found = tool_from_words(ws)
        if a["label"] == "tool_pick_up":
            tool_in_hand = found or tool_in_hand
            a["tool"] = tool_in_hand
        elif a["label"] == "tool_put_down":
            a["tool"] = tool_in_hand
            tool_in_hand = ""
        else:
            # gereedschap in de hand gaat voor; anders wat er in deze actie genoemd wordt (bijv. 'hand')
            a["tool"] = tool_in_hand or found
        a["transcript"] = tekst
    return actions, kept, dropped


# ---------------------------------------------------------------------------
# Excel-overzicht voor de spreker (annotatie + controle + transcript + samenvatting)
# ---------------------------------------------------------------------------
LABEL_NL = {
    "identify": "verbinding zoeken", "rotate_object": "omdraaien", "fixate": "vasthouden",
    "tool_pick_up": "gereedschap pakken", "tool_put_down": "gereedschap neerleggen",
    "tool_position": "gereedschap plaatsen", "screw_unscrew": "schroef losdraaien",
    "screw_tighten": "schroef vastdraaien", "pry_component": "loswrikken",
    "disconnect_other": "loskoppelen", "component_remove": "onderdeel verwijderen",
}


def mmss(t):
    if t == "" or t is None:
        return ""
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:04.1f}"


def transcript_lines(words, gap=1.0):
    lines, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > gap:
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [(l[0]["start"], l[-1]["end"], " ".join(w["word"] for w in l)) for l in lines]


def write_excel(path, actions, words, summary_rows, dropped, meta):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.formatting.rule import CellIsRule

    head_fill = PatternFill("solid", fgColor="1F4E79")
    head_font = Font(bold=True, color="FFFFFF")
    input_fill = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(top=thin, bottom=thin, left=thin, right=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    def header(ws, row, cols, widths):
        for i, (c, w) in enumerate(zip(cols, widths), start=1):
            cell = ws.cell(row=row, column=i, value=c)
            cell.fill, cell.font, cell.alignment, cell.border = head_fill, head_font, wrap, border
            ws.column_dimensions[cell.column_letter].width = w
        ws.freeze_panes = ws.cell(row=row + 1, column=1)

    wb = Workbook()

    # --- Tabblad 1: Annotatie -------------------------------------------------
    ws = wb.active
    ws.title = "Annotatie"
    ws["A1"] = f"Annotatie met spraak – {meta['opname']}"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E79")
    ws["A2"] = (f"Gemaakt op {meta['datum']} met {meta['modellen']}. "
                "Controleer elke regel en kies in kolom K of hij klopt (gele kolommen invullen).")
    cols = ["#", "Begin (m:ss)", "Eind (m:ss)", "Duur (s)", "Handeling", "Label", "eDIM-categorie",
            "Gereedschap", "Toelichting (wat je zei)", "Frames (30 fps)", "Klopt? (ja/nee)", "Opmerking"]
    header(ws, 4, cols, [5, 12, 12, 9, 22, 18, 16, 28, 50, 14, 14, 30])
    dv = DataValidation(type="list", formula1='"ja,nee"', allow_blank=True)
    ws.add_data_validation(dv)
    first = 5
    for k, a in enumerate(actions, start=1):
        r = first + k - 1
        dur = round(float(a["end_s"]) - float(a["start_s"]), 1) if a["end_s"] != "" else ""
        vals = [k, mmss(a["start_s"]), mmss(a["end_s"]), dur, LABEL_NL.get(a["label"], a["label"]),
                a["label"], a["edim"], a.get("tool", ""), a.get("transcript", ""),
                f"{a['start_frame']}–{a['end_frame']}", "", a.get("notes", "")]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.alignment, cell.border = wrap, border
        for c in (11, 12):
            ws.cell(row=r, column=c).fill = input_fill
        dv.add(f"K{r}")
    last = first + len(actions) - 1
    ws.conditional_formatting.add(f"K{first}:K{last}", CellIsRule(operator="equal", formula=['"ja"'],
                                  fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(f"K{first}:K{last}", CellIsRule(operator="equal", formula=['"nee"'],
                                  fill=PatternFill("solid", fgColor="FFC7CE")))

    # --- Tabblad 2: Bewijs (toegevoegde waarde) -------------------------------
    wb2 = wb.create_sheet("Toegevoegde waarde")
    wb2.column_dimensions["A"].width = 48
    wb2.column_dimensions["B"].width = 18
    wb2.column_dimensions["C"].width = 50
    wb2["A1"] = "Toegevoegde waarde van annoteren met spraak"
    wb2["A1"].font = Font(bold=True, size=14, color="1F4E79")
    rows = [
        ("Aantal handelingen gevonden door het model", f"=COUNTA(Annotatie!A{first}:A{last})", "automatisch"),
        ("Aantal gecontroleerd als goed (ja)", f'=COUNTIF(Annotatie!K{first}:K{last},"ja")', "telt kolom K"),
        ("Aantal gecontroleerd als fout (nee)", f'=COUNTIF(Annotatie!K{first}:K{last},"nee")', "telt kolom K"),
        ("Aantal handelingen dat het model miste", None, "zelf invullen"),
        ("Score: goed ÷ (goed + fout + gemist)", "=IFERROR(B4/(B4+B5+B6),\"\")", "automatisch"),
        ("", "", ""),
        ("Duur van de opname (minuten)", round(meta["duur_s"] / 60, 1), "automatisch"),
        ("Rekentijd van het model (minuten)", round(meta["rekentijd_s"] / 60, 1), "automatisch"),
        ("Tijd om deze lijst te controleren (minuten)", None, "zelf invullen"),
        ("Tijd voor handmatig annoteren van dezelfde opname (minuten)", None,
         "zelf invullen (schatting of gemeten)"),
        ("Tijd met spraak (rekentijd + controle)", "=B10+B11", "automatisch"),
        ("Tijdwinst (minuten)", '=IF(B12="","",B12-B13)', "automatisch"),
        ("Tijdwinst (%)", '=IFERROR((B12-B13)/B12,"")', "automatisch"),
    ]
    for i, (a, b, c) in enumerate(rows, start=3):
        wb2.cell(row=i, column=1, value=a)
        wb2.cell(row=i, column=2, value=b)
        wb2.cell(row=i, column=3, value=c).font = Font(italic=True, color="808080")
        if c == "zelf invullen" or c.startswith("zelf invullen"):
            wb2.cell(row=i, column=2).fill = input_fill
    wb2["B7"].number_format = "0%"
    wb2["B15"].number_format = "0%"
    # rows: 3 gevonden,4 ja,5 nee,6 gemist,7 nauwk.,9 duur,10 rekentijd,11 controle,12 handmatig,13 met spraak,14 winst,15 winst%
    wb2["B13"] = "=B10+B11"
    wb2["B14"] = '=IF(B12="","",B12-B13)'
    wb2["B15"] = '=IFERROR((B12-B13)/B12,"")'
    wb2["A17"] = "Uitleg"
    wb2["A17"].font = Font(bold=True)
    wb2["A18"] = ("Vul de gele vakjes in. De rest rekent Excel zelf uit. Zo laat je zien hoeveel procent het "
                  "model goed had en hoeveel tijd inspreken + controleren scheelt ten opzichte van handmatig annoteren.")
    wb2["A18"].alignment = wrap
    wb2.merge_cells("A18:C20")

    # --- Tabblad 3: Transcript ------------------------------------------------
    wt = wb.create_sheet("Transcript")
    header(wt, 1, ["Begin (m:ss)", "Eind (m:ss)", "Wat er gezegd is"], [12, 12, 100])
    for i, (s, e, t) in enumerate(transcript_lines(words), start=2):
        for c, v in enumerate([mmss(s), mmss(e), t], start=1):
            cell = wt.cell(row=i, column=c, value=v)
            cell.alignment, cell.border = wrap, border

    # --- Tabblad 4: Technische details ----------------------------------------
    wd = wb.create_sheet("Details")
    header(wd, 1, ["Model", "Rekentijd (s)", "RTF", "Woorden", "Commando's", "Acties"], [24, 14, 8, 10, 12, 10])
    for i, s in enumerate(summary_rows, start=2):
        for c, k in enumerate(["model", "seconds", "rtf", "words", "commands", "actions"], start=1):
            wd.cell(row=i, column=c, value=s.get(k, ""))
    r0 = len(summary_rows) + 4
    wd.cell(row=r0, column=1, value="Weggefilterde commando's (niet bevestigd door Whisper)").font = Font(bold=True)
    header_row = r0 + 1
    for c, h in enumerate(["Begin (m:ss)", "Commando", "Gehoord als"], start=1):
        cell = wd.cell(row=header_row, column=c, value=h)
        cell.fill, cell.font = head_fill, head_font
    for i, d in enumerate(dropped, start=header_row + 1):
        wd.cell(row=i, column=1, value=mmss(d["start"]))
        wd.cell(row=i, column=2, value=d["command"])
        wd.cell(row=i, column=3, value=d["heard"])

    wb.save(path)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="opname (m4a, wav, mp3 of mp4)")
    ap.add_argument("--models", default="vosk,small",
                    help="komma-lijst uit: vosk, vosk-vrij, tiny, base, small, medium, large-v3-turbo, "
                         "large-v2, large-v3 (default: vosk,small)")
    ap.add_argument("--vosk-model", default="models/vosk-model-small-nl-0.22")
    ap.add_argument("--no-hint", action="store_true", help="Whisper zonder woordenlijst als hint")
    ap.add_argument("--fps", type=float, default=30.0, help="fps van de video (voor framenummers)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--vanaf", type=float, default=0.0,
                    help="neem in de annotatie alleen acties mee die na dit aantal seconden beginnen "
                         "(bijv. --vanaf 88 om de testdelen van de opname over te slaan)")
    ap.add_argument("--combi", metavar="WHISPER", default="",
                    help="maak ook een gecombineerde annotatie: Vosk-tijden + controle/tekst van dit "
                         "Whisper-model (bijv. --combi small). Vosk en dat model worden dan automatisch gedraaid.")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out) / audio_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio inlezen: {audio_path}")
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  duur: {duration:.1f} s")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.combi:
        for m in ("vosk", args.combi):
            if m not in model_list:
                model_list.append(m)
    results = {}
    summary = []
    for name in model_list:
        tag = name if (name.startswith("vosk") or not args.no_hint) else f"{name}-zonderhint"
        print(f"\n=== {tag} ===")
        t0 = time.time()
        if name == "vosk":
            words = run_vosk(audio, args.vosk_model, use_grammar=True)
        elif name == "vosk-vrij":
            words = run_vosk(audio, args.vosk_model, use_grammar=False)
        else:
            words = run_whisper(audio, name, use_hint=not args.no_hint)
        secs = time.time() - t0

        commands = find_commands(words)
        actions = build_actions(commands, audio_path.name, args.fps)
        results[name] = (words, commands)

        write_csv(out_dir / f"{tag}_words.csv", words, ["start", "end", "word", "conf"])
        write_csv(out_dir / f"{tag}_commands.csv", commands, ["start", "end", "command", "heard", "score"])
        write_csv(out_dir / f"{tag}_actions.csv", actions,
                  ["video", "start_frame", "end_frame", "label", "edim", "start_s", "end_s", "notes"])

        print("  tekst:", " ".join(w["word"] for w in words)[:300])
        print(f"  {len(words)} woorden, {len(commands)} commando's, {len(actions)} acties")
        print(f"  rekentijd {secs:.1f} s  (RTF {secs / duration:.2f}; < 1 = sneller dan realtime)")
        summary.append({"model": tag, "seconds": round(secs, 1), "rtf": round(secs / duration, 2),
                        "words": len(words), "commands": len(commands), "actions": len(actions)})

    if args.combi:
        v_words, v_cmds = results["vosk"]
        w_words, w_cmds = results[args.combi]
        actions, kept, dropped = combine(v_cmds, w_cmds, w_words, audio_path.name, args.fps)
        actions = [a for a in actions if float(a["start_s"]) >= args.vanaf]
        tag = f"combi_vosk+{args.combi}"
        write_csv(out_dir / f"{tag}_actions.csv", actions,
                  ["video", "start_frame", "end_frame", "label", "edim", "tool",
                   "start_s", "end_s", "transcript", "notes"])
        write_csv(out_dir / f"{tag}_verworpen.csv", dropped, ["start", "end", "command", "heard", "score"])
        print(f"\n=== {tag} ===")
        print(f"  {len(kept)} commando's bevestigd door Whisper, {len(dropped)} verworpen, {len(actions)} acties")
        summary.append({"model": tag, "seconds": "", "rtf": "", "words": len(w_words),
                        "commands": len(kept), "actions": len(actions)})
        rekentijd = sum(float(s["seconds"]) for s in summary if s["model"] in ("vosk", args.combi))
        from datetime import date
        xlsx = out_dir / f"{audio_path.stem}_annotatie.xlsx"
        write_excel(xlsx, actions, [w for w in w_words if w["start"] >= args.vanaf], summary, dropped,
                    {"opname": audio_path.name, "datum": date.today().strftime("%d-%m-%Y"),
                     "modellen": f"Vosk + Whisper {args.combi}", "duur_s": duration, "rekentijd_s": rekentijd})
        print(f"  Excel-overzicht: {xlsx}")

    write_csv(out_dir / "summary.csv", summary, ["model", "seconds", "rtf", "words", "commands", "actions"])
    print(f"\nKlaar. Resultaten staan in {out_dir}")


if __name__ == "__main__":
    main()
