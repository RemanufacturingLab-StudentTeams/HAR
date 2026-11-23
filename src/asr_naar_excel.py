"""Zet de resultaten van de spraakmodellen om naar één Excel-bestand.

Leest alle csv-bestanden uit een resultatenmap (bijv. outputs/opname_3min) en maakt
<map>/<opname>_resultaten.xlsx met:
  - Uitleg              wat er in het bestand staat
  - Overzicht           per model: rekentijd, aantal woorden, commando's en acties
  - Acties <model>      per handeling: tijd, handeling, eDIM, gereedschap, wat er gezegd is,
                        + gele kolom "Klopt?" (ja/nee) om te controleren
  - Transcript <model>  alles wat het model heeft verstaan, per zin met tijd
  - Weggefilterd        commando's die de combinatie heeft verworpen (als die er zijn)
  - Toegevoegde waarde  score en tijdwinst, rekent zelf uit na invullen

Gebruik (vanuit de projectmap):
    run_asr_excel.bat outputs\\opname_3min
    run_asr_excel.bat outputs\\opname_3min --vanaf 88     (alleen acties na 1:28)
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError:
    sys.exit("openpyxl ontbreekt. Installeer het eerst met:  python -m pip install openpyxl")

HANDELING = {
    "identify": "verbinding zoeken", "rotate_object": "omdraaien", "fixate": "vasthouden",
    "tool_pick_up": "gereedschap pakken", "tool_put_down": "gereedschap neerleggen",
    "tool_position": "gereedschap plaatsen", "screw_unscrew": "schroef losdraaien",
    "screw_tighten": "schroef vastdraaien", "pry_component": "loswrikken",
    "disconnect_other": "loskoppelen", "component_remove": "onderdeel verwijderen",
}
MODELNAAM = {"vosk": "Vosk", "vosk-vrij": "Vosk (vrij)", "small": "Whisper small",
             "medium": "Whisper medium", "large-v3-turbo": "Whisper turbo",
             "large-v2": "Whisper large-v2", "large-v3": "Whisper large-v3"}

BLAUW = PatternFill("solid", fgColor="1F4E79")
GEEL = PatternFill("solid", fgColor="FFF2CC")
GROEN = PatternFill("solid", fgColor="C6EFCE")
ROOD = PatternFill("solid", fgColor="FFC7CE")
WIT_VET = Font(bold=True, color="FFFFFF")
TITEL = Font(bold=True, size=14, color="1F4E79")
LIJN = Side(style="thin", color="BFBFBF")
RAND = Border(top=LIJN, bottom=LIJN, left=LIJN, right=LIJN)
WRAP = Alignment(wrap_text=True, vertical="top")


# ---------------------------------------------------------------------------
def lees_csv(pad):
    with open(pad, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f)]


def getal(x, standaard=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return standaard


def mmss(t):
    t = getal(t)
    if t is None:
        return ""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def model_naam(tag):
    if tag.startswith("combi_vosk+"):
        return "Combinatie Vosk + " + MODELNAAM.get(tag.split("+", 1)[1], tag.split("+", 1)[1])
    return MODELNAAM.get(tag, tag)


def kopregel(ws, rij, kolommen, breedtes):
    for i, (k, b) in enumerate(zip(kolommen, breedtes), start=1):
        c = ws.cell(row=rij, column=i, value=k)
        c.fill, c.font, c.alignment, c.border = BLAUW, WIT_VET, WRAP, RAND
        ws.column_dimensions[c.column_letter].width = b
    ws.freeze_panes = ws.cell(row=rij + 1, column=1)


def zinnen(woorden, pauze=1.0):
    """Groepeer losse woorden tot zinnen: een nieuwe zin na een pauze van > 1 s."""
    regels, huidig = [], []
    for w in woorden:
        s, e = getal(w.get("start")), getal(w.get("end"))
        if s is None:
            continue
        if huidig and s - huidig[-1][1] > pauze:
            regels.append(huidig)
            huidig = []
        huidig.append((s, e, w.get("word", "")))
    if huidig:
        regels.append(huidig)
    return [(r[0][0], r[-1][1], " ".join(x[2] for x in r)) for r in regels]


def tekst_tussen(woorden, t0, t1):
    if not woorden or t0 is None:
        return ""
    t1 = t1 if t1 is not None else t0 + 10
    return " ".join(w["word"] for w in woorden
                    if getal(w.get("start")) is not None
                    and t0 - 1.0 <= (getal(w["start"]) + getal(w["end"], getal(w["start"]))) / 2 <= t1)


# ---------------------------------------------------------------------------
def blad_acties(wb, titel, acties, woorden, opname, modeltekst):
    ws = wb.create_sheet(titel[:31])
    ws["A1"] = f"{modeltekst} – {opname}"
    ws["A1"].font = TITEL
    ws["A2"] = "Controleer elke regel: luister de opname terug en kies in de gele kolom 'ja' of 'nee'."
    kol = ["#", "Begin (m:ss)", "Eind (m:ss)", "Duur (s)", "Handeling", "eDIM-categorie",
           "Gereedschap", "Wat er gezegd is", "Label", "Frames (30 fps)", "Klopt? (ja/nee)",
           "Opmerking", "Melding van het model"]
    kopregel(ws, 4, kol, [5, 12, 12, 9, 22, 16, 26, 50, 18, 14, 14, 28, 26])
    dv = DataValidation(type="list", formula1='"ja,nee"', allow_blank=True)
    ws.add_data_validation(dv)
    eerste = 5
    for k, a in enumerate(acties, start=1):
        r = eerste + k - 1
        s, e = getal(a.get("start_s")), getal(a.get("end_s"))
        tekst = tekst_tussen(woorden, s, e) if woorden else a.get("transcript", "")
        waarden = [k, mmss(s), mmss(e), round(e - s, 1) if (s is not None and e is not None) else "",
                   HANDELING.get(a.get("label", ""), a.get("label", "")), a.get("edim", ""),
                   a.get("tool", ""), tekst, a.get("label", ""),
                   f"{a.get('start_frame', '')}–{a.get('end_frame', '')}", "", "", a.get("notes", "")]
        for c, v in enumerate(waarden, start=1):
            cel = ws.cell(row=r, column=c, value=v)
            cel.alignment, cel.border = WRAP, RAND
        ws.cell(row=r, column=11).fill = GEEL
        ws.cell(row=r, column=12).fill = GEEL
        dv.add(f"K{r}")
    laatste = max(eerste, eerste + len(acties) - 1)
    bereik = f"K{eerste}:K{laatste}"
    ws.conditional_formatting.add(bereik, CellIsRule(operator="equal", formula=['"ja"'], fill=GROEN))
    ws.conditional_formatting.add(bereik, CellIsRule(operator="equal", formula=['"nee"'], fill=ROOD))
    return ws.title, eerste, laatste


def blad_transcript(wb, titel, woorden):
    ws = wb.create_sheet(titel[:31])
    kopregel(ws, 1, ["Begin (m:ss)", "Eind (m:ss)", "Wat het model verstond"], [12, 12, 100])
    for i, (s, e, t) in enumerate(zinnen(woorden), start=2):
        for c, v in enumerate([mmss(s), mmss(e), t], start=1):
            cel = ws.cell(row=i, column=c, value=v)
            cel.alignment, cel.border = WRAP, RAND


def blad_waarde(wb, bladnaam, eerste, laatste, duur_s, rekentijd_s):
    ws = wb.create_sheet("Toegevoegde waarde")
    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 40
    ws["A1"] = "Toegevoegde waarde van annoteren met spraak"
    ws["A1"].font = TITEL
    ref = f"'{bladnaam}'!"
    rijen = [
        (3, "Handelingen gevonden door het model", f"=COUNTA({ref}A{eerste}:A{laatste})", "automatisch"),
        (4, "Gecontroleerd als goed (ja)", f'=COUNTIF({ref}K{eerste}:K{laatste},"ja")', "automatisch uit 'Klopt?'"),
        (5, "Gecontroleerd als fout (nee)", f'=COUNTIF({ref}K{eerste}:K{laatste},"nee")', "automatisch uit 'Klopt?'"),
        (6, "Handelingen die het model miste", None, "zelf invullen"),
        (7, "Score: goed ÷ (goed + fout + gemist)", '=IFERROR(B4/(B4+B5+B6),"")', "automatisch"),
        (9, "Duur van de opname (minuten)", round(duur_s / 60, 1) if duur_s else None,
         "automatisch" if duur_s else "zelf invullen"),
        (10, "Rekentijd van het model (minuten)", round(rekentijd_s / 60, 1) if rekentijd_s else None,
         "automatisch" if rekentijd_s else "zelf invullen"),
        (11, "Tijd om de lijst te controleren (minuten)", None, "zelf invullen"),
        (12, "Tijd voor handmatig annoteren van dezelfde opname (minuten)", None, "zelf invullen"),
        (13, "Tijd met spraak (rekentijd + controle)", '=IF(B11="","",B10+B11)', "automatisch"),
        (14, "Tijdwinst (minuten)", '=IF(OR(B12="",B13=""),"",B12-B13)', "automatisch"),
        (15, "Tijdwinst (%)", '=IFERROR((B12-B13)/B12,"")', "automatisch"),
    ]
    for r, a, b, c in rijen:
        ws.cell(row=r, column=1, value=a)
        ws.cell(row=r, column=2, value=b)
        ws.cell(row=r, column=3, value=c).font = Font(italic=True, color="808080")
        if c.startswith("zelf"):
            ws.cell(row=r, column=2).fill = GEEL
    ws["B7"].number_format = "0%"
    ws["B15"].number_format = "0%"
    ws["A17"] = "Uitleg"
    ws["A17"].font = Font(bold=True)
    ws["A18"] = (f"De score en tijdwinst gaan over het blad '{bladnaam}'. Vul de gele vakjes in; "
                 "Excel rekent de rest uit.")
    ws["A18"].alignment = WRAP
    ws.merge_cells("A18:C19")


def blad_uitleg(ws, opname, modellen, vanaf):
    ws.title = "Uitleg"
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 90
    ws["A1"] = f"Resultaten spraakmodellen – {opname}"
    ws["A1"].font = TITEL
    regels = [
        ("Gemaakt op", date.today().strftime("%d-%m-%Y")),
        ("Modellen", ", ".join(modellen)),
        ("Wat is dit?", "De spreker heeft tijdens het demonteren hardop commando's ingesproken "
                        "(bijv. 'start losdraaien' … 'klaar'). De spraakmodellen hebben de opname "
                        "uitgeschreven en daar automatisch een annotatie van gemaakt."),
        ("Overzicht", "Per model: hoe lang het rekende en hoeveel woorden, commando's en handelingen het vond."),
        ("Acties …", "Per model de gevonden handelingen. Controleer in de gele kolom 'Klopt?' of ze kloppen."),
        ("Transcript …", "Alles wat het model heeft verstaan, per zin met tijd."),
        ("Toegevoegde waarde", "Score en tijdwinst: rekent automatisch uit zodra je de gele vakjes invult."),
    ]
    if vanaf:
        regels.append(("Let op", f"Alleen handelingen vanaf {mmss(vanaf)} zijn meegenomen; "
                                 "de testdelen daarvoor zijn weggelaten."))
    for i, (a, b) in enumerate(regels, start=3):
        ws.cell(row=i, column=1, value=a).font = Font(bold=True)
        ws.cell(row=i, column=2, value=b).alignment = WRAP


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="resultatenmap, bijv. outputs/opname_3min")
    ap.add_argument("--vanaf", type=float, default=0.0,
                    help="alleen acties die na dit aantal seconden beginnen (bijv. 88)")
    ap.add_argument("--uit", default="", help="naam van het Excel-bestand (optioneel)")
    args = ap.parse_args()

    map_ = Path(args.map)
    if not map_.is_dir():
        sys.exit(f"Map niet gevonden: {map_}")
    opname = map_.name

    actie_bestanden = sorted(map_.glob("*_actions.csv"))
    if not actie_bestanden:
        sys.exit(f"Geen *_actions.csv gevonden in {map_}. Draai eerst run_asr_test.bat.")
    tags = [p.name[:-len("_actions.csv")] for p in actie_bestanden]
    # combinatie eerst, daarna de losse modellen
    tags.sort(key=lambda t: (not t.startswith("combi"), t))

    samenvatting = lees_csv(map_ / "summary.csv") if (map_ / "summary.csv").exists() else []
    woorden = {t: lees_csv(map_ / f"{t}_words.csv") for t in tags if (map_ / f"{t}_words.csv").exists()}

    wb = Workbook()
    blad_uitleg(wb.active, opname, [model_naam(t) for t in tags], args.vanaf)

    # Overzicht
    ov = wb.create_sheet("Overzicht")
    kopregel(ov, 1, ["Model", "Rekentijd (s)", "RTF (< 1 = sneller dan realtime)", "Woorden",
                     "Commando's", "Acties"], [34, 14, 30, 10, 12, 10])
    for i, s in enumerate(samenvatting, start=2):
        vals = [model_naam(s.get("model", "")), getal(s.get("seconds"), ""), getal(s.get("rtf"), ""),
                getal(s.get("words"), ""), getal(s.get("commands"), ""), getal(s.get("actions"), "")]
        for c, v in enumerate(vals, start=1):
            ov.cell(row=i, column=c, value=v).border = RAND

    # Acties + transcript per model
    hoofd = None
    for t in tags:
        acties = [a for a in lees_csv(map_ / f"{t}_actions.csv")
                  if getal(a.get("start_s"), 0) >= args.vanaf]
        w = woorden.get(t)
        if w is None and t.startswith("combi_vosk+"):
            w = woorden.get(t.split("+", 1)[1])
        kort = "Combinatie" if t.startswith("combi") else model_naam(t)
        info = blad_acties(wb, f"Acties {kort}", acties, w or [], opname, model_naam(t))
        if hoofd is None:
            hoofd = info
    for t in tags:
        if t in woorden:
            blad_transcript(wb, f"Transcript {model_naam(t)}",
                            [x for x in woorden[t] if getal(x.get("start"), 0) >= args.vanaf])

    # Weggefilterde commando's
    for p in map_.glob("*_verworpen.csv"):
        ws = wb.create_sheet("Weggefilterd")
        kopregel(ws, 1, ["Begin (m:ss)", "Commando", "Gehoord als"], [12, 24, 30])
        for i, d in enumerate(lees_csv(p), start=2):
            ws.cell(row=i, column=1, value=mmss(d.get("start")))
            ws.cell(row=i, column=2, value=d.get("command", ""))
            ws.cell(row=i, column=3, value=d.get("heard", ""))
        break

    # Toegevoegde waarde (over het eerste actieblad: de combinatie als die er is)
    duur = None
    alle_woorden = [x for ws_ in woorden.values() for x in ws_]
    if alle_woorden:
        duur = max(getal(x.get("end"), 0) for x in alle_woorden)
    def rekentijd_van(model):
        return next((getal(s.get("seconds"), 0) for s in samenvatting if s.get("model") == model), 0)
    eerste_tag = tags[0]
    if eerste_tag.startswith("combi_vosk+"):
        rekentijd = rekentijd_van("vosk") + rekentijd_van(eerste_tag.split("+", 1)[1])
    else:
        rekentijd = rekentijd_van(eerste_tag)
    blad_waarde(wb, hoofd[0], hoofd[1], hoofd[2], duur, rekentijd)

    uit = map_ / (args.uit or f"{opname}_resultaten.xlsx")
    try:
        wb.save(uit)
    except PermissionError:
        sys.exit(f"Kan {uit} niet opslaan. Is het bestand nog open in Excel? Sluit het en probeer opnieuw.")
    print(f"Excel-bestand gemaakt: {uit}")
    print(f"  bladen: {', '.join(wb.sheetnames)}")


if __name__ == "__main__":
    main()
