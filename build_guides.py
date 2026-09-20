#!/usr/bin/env python3
"""
build_guides.py — Re-runnable pipeline: Wikivoyage -> md/<city>.md budget travel guides.

Refreshes the chatbot's travel knowledge base from Wikivoyage (free, public,
no API key needed). Big cities automatically merge their district sub-articles
(e.g. Tokyo/Shinjuku) so listings are captured from every district.

Usage:
    python build_guides.py                     # fill in MISSING cities only (safe to re-run;
                                                 #   use this to resume after rate-limit failures)
    python build_guides.py --all                # force-refresh ALL cities (~15 min, may hit
                                                 #   rate limits — just re-run to fill the gaps)
    python build_guides.py --city tokyo osaka    # refresh specific cities
    python build_guides.py --dump enwikivoyage-latest-pages-articles.xml.bz2
                                                # parse a local XML dump instead of the API
                                                #   (no rate limits; download from
                                                #   dumps.wikimedia.org)

Each generated md/<city>.md contains (targeted at budget travellers):
  - Commute tips (from the "Get around" section)
  - Sightseeing, split into categories: History / Art / Culture / Sports / Nature / Other
  - Budget eats (budget & mid-range subsections; "Splurge" is dropped)
  - Budget stays (hostels & cheap accommodation from "Sleep")
  - Budget drinks (bars & cheap nightlife from "Drink")
  - External links (restaurant/hotel/museum websites) where Wikivoyage provides them

After refreshing, restart the chatbot server (python app.py) — the BM25
retrieval index is rebuilt from md/ at startup.

Content is CC BY-SA (Wikivoyage); attribution line is included in each file.
"""

import bz2
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

API = "https://en.wikivoyage.org/w/api.php"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md")

# slug -> Wikivoyage article title
CITIES = {
    # --- Asia-Pacific destinations ---
    # Japan
    "tokyo": "Tokyo", "osaka": "Osaka", "nagoya": "Nagoya", "fukuoka": "Fukuoka",
    "sapporo": "Sapporo", "okinawa": "Okinawa", "kagoshima": "Kagoshima",
    "sendai": "Sendai", "hiroshima": "Hiroshima", "kumamoto": "Kumamoto",
    "matsuyama": "Matsuyama", "ishigaki": "Ishigaki", "miyazaki": "Miyazaki",
    "takamatsu": "Takamatsu", "niigata": "Niigata", "aomori": "Aomori",
    # Korea
    "seoul": "Seoul", "busan": "Busan", "daegu": "Daegu",
    # Taiwan
    "taipei": "Taipei", "kaohsiung": "Kaohsiung", "taichung": "Taichung",
    # China
    "beijing": "Beijing", "shanghai": "Shanghai", "hangzhou": "Hangzhou", "ningbo": "Ningbo",
    # Thailand
    "bangkok": "Bangkok", "chiang-mai": "Chiang Mai", "phuket": "Phuket", "krabi": "Krabi",
    # Vietnam / Cambodia
    "da-nang": "Da Nang", "phnom-penh": "Phnom Penh", "siem-reap": "Siem Reap",
    # SE Asia & Pacific
    "singapore": "Singapore", "penang": "Penang", "kota-kinabalu": "Kota Kinabalu",
    "manila": "Manila", "cebu": "Cebu City", "iloilo": "Iloilo City",
    "guam": "Guam", "saipan": "Saipan",
    # --- Major European cities ---
    "london": "London", "paris": "Paris", "berlin": "Berlin", "rome": "Rome",
    "amsterdam": "Amsterdam", "barcelona": "Barcelona", "madrid": "Madrid",
    "vienna": "Vienna", "prague": "Prague", "budapest": "Budapest", "lisbon": "Lisbon",
    "munich": "Munich", "milan": "Milan", "venice": "Venice", "florence": "Florence",
    "copenhagen": "Copenhagen", "stockholm": "Stockholm", "athens": "Athens",
    "dublin": "Dublin", "zurich": "Zurich", "brussels": "Brussels",
}

# ---------------------------------------------------------------------------
# Wikitext parsing helpers
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


BULLET_RE = re.compile(r"^\*\s+'''(.+?)'''[.:—-]?\s*(.*)$", re.M)


def find_listing_templates(text):
    """Find {{listing|...}} / {{see|...}} / {{do|...}} / {{eat|...}} / {{drink|...}}
    / {{sleep|...}} templates, correctly handling nested {{...}} inside parameters."""
    results = []
    for m in re.finditer(r"\{\{\s*(listing|see|do|eat|drink|sleep)\s*\|", text, re.I):
        start = m.end()  # position right after the opening "|"
        depth = 1        # we are inside one {{ ... }}
        i = start
        while i < len(text) and depth > 0:
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
            else:
                i += 1
        if depth == 0:
            # body is between start and the final "}}" (exclusive)
            results.append(text[start:i - 2])
    return results


# Sightseeing categorisation (checked in priority order)
CATEGORIES = [
    ("History", re.compile(
        r"temple|shrine|castle|ruins?\b|monument|memorial|historic|fort\b|palace|"
        r"mosque|church|cathedral|tomb|archaeolog|war|battle|samurai|shogun|edo\b|meiji", re.I)),
    ("Art", re.compile(
        r"museum|gallery|art\b|sculpture|painting|exhibition", re.I)),
    ("Sports", re.compile(
        r"stadium|sport|football|soccer|baseball|sumo|rugby|basketball|hockey|"
        r"marathon|arena|match|cycling|skiing|surfing|diving|snorkel", re.I)),
    ("Nature", re.compile(
        r"park|garden|mountain|beach|island|lake|river|waterfall|nature|zoo|"
        r"aquarium|onsen|hot spring|viewpoint|observation|forest|trail|hike", re.I)),
    ("Culture", re.compile(
        r"theatre|theater|festival|cultur|traditional|performance|opera|concert|"
        r"dance|market|geisha|tea ceremony|neighbourhood|neighborhood|district|"
        r"temple town|old town|chinatown", re.I)),
]


def get_sections(wikitext):
    """Split wikitext into (level, title, body) sections."""
    matches = list(HEADER_RE.finditer(wikitext))
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        sections.append((len(m.group(1)), m.group(2).strip(), wikitext[start:end]))
    return sections


def find_section(sections, title, level=2):
    """Return the bodies of ALL level-2 sections with this title (the main
    article + every district article), including their subsections.
    Subsections are only captured while inside a matching section — the
    capture stops at the next non-matching level-2 header."""
    parts = []
    in_target = False
    for lvl, ttl, body in sections:
        if lvl == level:
            if ttl.lower() == title.lower():
                in_target = True
                parts.append(body)
            else:
                in_target = False
        elif in_target and lvl > level:
            # subsection of the currently-matched section only
            parts.append(f"\n### {ttl}\n{body}")
    return "\n".join(parts) if parts else None


def clean_wikitext(text, max_chars=2200):
    """Convert wikitext to readable plain text/markdown."""
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\[\[Image:[^\]]*\]\]", "", text)      # [[Image:...]] blocks
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text)       # [[File:...]] blocks
    text = re.sub(r"^thumb\|[^\n]*$", "", text, flags=re.M)  # leftover thumb params
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)          # simple templates
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)           # second pass for nesting
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)   # [[link|text]]
    text = re.sub(r"\[([^\s\]]+)\s+([^\]]+)\]", r"[\2](\1)", text)  # [url text]
    text = re.sub(r"'{2,3}", "", text)                    # bold/italic
    text = re.sub(r"^\s*[:*#]+\s*", "- ", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


def parse_listing(body):
    """Parse a Wikivoyage listing template body into a params dict."""
    params = {}
    positional = []
    for part in body.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip().lower()] = v.strip()
        elif part.strip():
            positional.append(part.strip())
    if "name" not in params and positional:
        params["name"] = positional[0]
    return params


def extract_listings(section_body):
    """Extract listings from a section: both {{listing}} templates AND
    plain bullet items in the '* '''Name''' description' style."""
    listings = [parse_listing(body) for body in find_listing_templates(section_body)]

    # Plain bullet style: * '''Name''' — description
    for m in BULLET_RE.finditer(section_body):
        name, desc = m.group(1).strip(), m.group(2).strip()
        # skip image/file links and navigation cruft
        if name.lower().startswith(("image", "file:")):
            continue
        # extract an external link from the description if present
        url_m = re.search(r"\[([https]?://[^\s\]]+)\s+[^\]]*\]", desc)
        params = {"name": name, "content": desc}
        if url_m:
            params["url"] = url_m.group(1)
        listings.append(params)
    return listings


def categorize(name, content):
    text = f"{name} {content}"
    for cat, rx in CATEGORIES:
        if rx.search(text):
            return cat
    return "Other"


def listing_line(params):
    name = params.get("name", "Unnamed")
    content = clean_wikitext(params.get("content", ""), max_chars=300)
    url = params.get("url", "").strip()
    price = params.get("price", "").strip()
    line = f"- **{name}**"
    if price:
        line += f" ({clean_wikitext(price, 80)})"
    if content:
        line += f" — {content}"
    if url:
        line += f" [Website]({url})"
    return line


# ---------------------------------------------------------------------------
# Guide generation
# ---------------------------------------------------------------------------

def build_guide(city_label, wikitext, district_wikitexts=None):
    """Build the guide from the main article plus optional district sub-articles."""
    all_sections = get_sections(wikitext)
    for _, dt in (district_wikitexts or []):
        all_sections += get_sections(dt)
    sections = all_sections
    out = [f"# {city_label} — Budget Travel Guide", ""]
    out.append("> Curated from [Wikivoyage](https://en.wikivoyage.org) (CC BY-SA), "
               "filtered for budget travellers. Prices may change — verify before you go.")
    out.append("")

    # --- Commute tips ---
    get_around = find_section(sections, "Get around")
    out.append("## Commute Tips")
    if get_around:
        out.append(clean_wikitext(get_around))
    else:
        out.append("(No commute information available in the guide.)")
    out.append("")

    # --- Sightseeing, categorised ---
    see_body = find_section(sections, "See") or ""
    do_body = find_section(sections, "Do") or ""
    listings = extract_listings(see_body) + extract_listings(do_body)

    out.append("## Sightseeing")
    if not listings:
        out.append("(No sightseeing listings found — see the Wikivoyage article directly.)")
    else:
        by_cat = {}
        for p in listings:
            if not p.get("name"):
                continue
            cat = categorize(p["name"], p.get("content", ""))
            by_cat.setdefault(cat, []).append(listing_line(p))
        for cat in ["History", "Art", "Culture", "Sports", "Nature", "Other"]:
            if cat in by_cat:
                out.append(f"\n### {cat}")
                out.extend(by_cat[cat][:30])  # cap per category
    out.append("")

    # --- Budget eats ---
    out.append("## Budget Eats")
    eat_body = find_section(sections, "Eat")
    budget_lines = []
    if eat_body:
        eat_sections = get_sections(eat_body)
        for sub_lvl, sub_ttl, sub_body in eat_sections:
            t = sub_ttl.lower()
            if "budget" in t or "cheap" in t or "mid-range" in t or "midrange" in t:
                for p in extract_listings(sub_body):
                    if p.get("name"):
                        budget_lines.append(listing_line(p))
        if not budget_lines:
            # fall back to the first few Eat listings (skip Splurge)
            for p in extract_listings(eat_body)[:8]:
                if p.get("name"):
                    budget_lines.append(listing_line(p))
    if budget_lines:
        out.extend(budget_lines[:30])
    else:
        out.append("(No budget restaurant listings found — ask a local or check "
                   "the Wikivoyage article.)")
    out.append("")

    # --- Budget stays (hostels & cheap accommodation) ---
    out.append("## Budget Stays")
    sleep_body = find_section(sections, "Sleep")
    stay_lines = []
    if sleep_body:
        sleep_sections = get_sections(sleep_body)
        for sub_lvl, sub_ttl, sub_body in sleep_sections:
            t = sub_ttl.lower()
            if any(k in t for k in ("budget", "hostel", "cheap", "mid-range", "midrange", "camping")):
                for p in extract_listings(sub_body):
                    if p.get("name"):
                        stay_lines.append(listing_line(p))
        if not stay_lines:
            # Fallback: listings whose price mentions cheap currencies/hostels
            # or that are hostels; skip obvious luxury (5-star, 'luxury', high prices)
            for p in extract_listings(sleep_body):
                if not p.get("name"):
                    continue
                blob = f"{p.get('name', '')} {p.get('price', '')} {p.get('content', '')}".lower()
                if "hostel" in blob or "guesthouse" in blob or "guest house" in blob:
                    stay_lines.append(listing_line(p))
                elif any(k in blob for k in ("luxury", "5-star", "five-star")):
                    continue
                else:
                    stay_lines.append(listing_line(p))
                if len(stay_lines) >= 30:
                    break
    if stay_lines:
        out.extend(stay_lines[:30])
    else:
        out.append("(No budget accommodation listings found — ask a local or check "
                   "the Wikivoyage article.)")
    out.append("")

    # --- Budget drinks (bars & cheap nightlife) ---
    out.append("## Budget Drinks")
    drink_body = find_section(sections, "Drink")
    drink_lines = []
    if drink_body:
        drink_sections = get_sections(drink_body)
        for sub_lvl, sub_ttl, sub_body in drink_sections:
            t = sub_ttl.lower()
            if "budget" in t or "cheap" in t or "bar" in t:
                for p in extract_listings(sub_body):
                    if p.get("name"):
                        drink_lines.append(listing_line(p))
        if not drink_lines:
            for p in extract_listings(drink_body)[:10]:
                if p.get("name"):
                    drink_lines.append(listing_line(p))
    if drink_lines:
        out.extend(drink_lines[:30])
    else:
        out.append("(No budget bar/nightlife listings found — ask a local or check "
                   "the Wikivoyage article.)")
    out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Content sources: API (default) or local XML dump
# ---------------------------------------------------------------------------

def fetch_via_api(title):
    for attempt in range(4):
        try:
            r = requests.get(API, params={
                "action": "parse", "page": title, "prop": "wikitext",
                "format": "json", "redirects": 1,
            }, timeout=60, headers={"User-Agent": "TravelGuideBuilder/1.0 (personal project)"})
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  rate-limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            data = r.json()
            if "error" in data:
                return None
            return data["parse"]["wikitext"]["*"]
        except requests.exceptions.JSONDecodeError:
            wait = 5 * (attempt + 1)
            print(f"  bad response for {title}, retry in {wait}s ...")
            time.sleep(wait)
    return None


def fetch_districts(city_title, wikitext):
    """Fetch the district sub-articles (e.g. 'Tokyo/Shinjuku') for a big city.

    Returns a list of (district_name, wikitext) for districts that exist.
    """
    # District links appear as [[City/District|...]] in the article
    # (also handles [[City/District#Anchor]] and [[City/District#Anchor|text]])
    district_links = sorted(set(re.findall(
        r"\[\[" + re.escape(city_title) + r"/([A-Za-z0-9 .'\-]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]",
        wikitext)))
    results = []
    for d in district_links[:12]:  # cap to avoid hammering the API
        full = f"{city_title}/{d}"
        try:
            dt = fetch_via_api(full)
            time.sleep(0.8)
        except Exception:
            dt = None
        if dt:
            results.append((d, dt))
            print(f"    district: {full}")
    return results


def fetch_from_dump(dump_path, wanted_titles):
    """Stream-parse a Wikivoyage XML dump, returning {title: wikitext}."""
    found = {}
    wanted = {t.lower(): t for t in wanted_titles}
    opener = bz2.open if dump_path.endswith(".bz2") else open
    with opener(dump_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag.endswith("}page"):
                title_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.10/}title")
                text_el = elem.find(".//{http://www.mediawiki.org/xml/export-0.10/}text")
                if title_el is not None and text_el is not None:
                    t = title_el.text.strip()
                    if t.lower() in wanted:
                        found[wanted[t.lower()]] = text_el.text or ""
                        print(f"  dump: found {t}")
                elem.clear()
    return found


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dump_path = None
    only_slugs = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dump":
            dump_path = args[i + 1]
            i += 2
        elif args[i] == "--city":
            # --city tokyo osaka   OR   --city tokyo --city osaka
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                only_slugs.append(args[i].lower())
                i += 1
        else:
            i += 1

    # --all / --force: regenerate every city even if the file exists.
    # --city implies regenerating even if the file exists.
    force = only_slugs or ("--all" in args) or ("--force" in args)

    cities = CITIES
    if only_slugs:
        cities = {s: CITIES[s] for s in only_slugs if s in CITIES}
        unknown = [s for s in only_slugs if s not in CITIES]
        if unknown:
            print(f"Unknown city slug(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(CITIES))}")
            return

    dump_cache = {}
    if dump_path:
        print(f"Parsing dump: {dump_path} ...")
        dump_cache = fetch_from_dump(dump_path, CITIES.values())

    ok, failed = 0, []
    for slug, title in cities.items():
        out_path = os.path.join(OUT_DIR, f"{slug}.md")
        if os.path.exists(out_path) and not force:
            print(f"skip {slug} (exists)")
            ok += 1
            continue

        if dump_path:
            wikitext = dump_cache.get(title)
        else:
            try:
                wikitext = fetch_via_api(title)
                time.sleep(1.0)  # be polite to the API
            except Exception as e:
                print(f"FAIL {title}: {type(e).__name__}")
                failed.append(title)
                continue

        if not wikitext:
            print(f"FAIL {title}: article not found")
            failed.append(title)
            continue

        # Big cities keep listings in district sub-articles (e.g. Tokyo/Shinjuku)
        districts = []
        try:
            districts = fetch_districts(title, wikitext)
        except Exception as e:
            print(f"  (district fetch failed for {title}: {type(e).__name__})")

        label = title.split(" (")[0]
        guide = build_guide(label, wikitext, districts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(guide)
        print(f"wrote md/{slug}.md")
        ok += 1

    print(f"\nDone: {ok} written, {len(failed)} failed.")
    if failed:
        print("Failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
