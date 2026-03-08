# =============================================================
# stock_search.py — NSE Stock Search & Ticker Resolution
# =============================================================
#
# How it works:
#   1. We maintain a local dictionary of 100+ popular NSE stocks
#      (name → ticker) so common searches are instant.
#   2. For anything not in the local dict, we use yfinance's
#      search API to look it up live.
#   3. The result is always a clean "SYMBOL.NS" ticker that the
#      rest of the app can use directly.
# =============================================================

import re
import streamlit as st
import yfinance as yf
from difflib import get_close_matches


# ─────────────────────────────────────────────────────────────
# Local NSE Stock Database  (name → ticker)
# Covers Nifty 50 + Nifty Next 50 + popular mid-caps
# ─────────────────────────────────────────────────────────────

NSE_STOCKS: dict[str, str] = {
    # ── Nifty 50 ──────────────────────────────────────────────
    "reliance industries":          "RELIANCE.NS",
    "reliance":                     "RELIANCE.NS",
    "tata consultancy services":    "TCS.NS",
    "tcs":                          "TCS.NS",
    "infosys":                      "INFY.NS",
    "infy":                         "INFY.NS",
    "hdfc bank":                    "HDFCBANK.NS",
    "hdfcbank":                     "HDFCBANK.NS",
    "icici bank":                   "ICICIBANK.NS",
    "icicibank":                    "ICICIBANK.NS",
    "wipro":                        "WIPRO.NS",
    "axis bank":                    "AXISBANK.NS",
    "axisbank":                     "AXISBANK.NS",
    "kotak mahindra bank":          "KOTAKBANK.NS",
    "kotak bank":                   "KOTAKBANK.NS",
    "kotakbank":                    "KOTAKBANK.NS",
    "state bank of india":          "SBIN.NS",
    "sbi":                          "SBIN.NS",
    "bharti airtel":                "BHARTIARTL.NS",
    "airtel":                       "BHARTIARTL.NS",
    "hindustan unilever":           "HINDUNILVR.NS",
    "hul":                          "HINDUNILVR.NS",
    "itc":                          "ITC.NS",
    "larsen and toubro":            "LT.NS",
    "l&t":                          "LT.NS",
    "lt":                           "LT.NS",
    "asian paints":                 "ASIANPAINT.NS",
    "maruti suzuki":                "MARUTI.NS",
    "maruti":                       "MARUTI.NS",
    "sun pharmaceutical":           "SUNPHARMA.NS",
    "sun pharma":                   "SUNPHARMA.NS",
    "sunpharma":                    "SUNPHARMA.NS",
    "titan":                        "TITAN.NS",
    "titan company":                "TITAN.NS",
    "bajaj finance":                "BAJFINANCE.NS",
    "bajajfinance":                 "BAJFINANCE.NS",
    "bajaj finserv":                "BAJAJFINSV.NS",
    "hcl technologies":             "HCLTECH.NS",
    "hcl tech":                     "HCLTECH.NS",
    "hcltech":                      "HCLTECH.NS",
    "tech mahindra":                "TECHM.NS",
    "techm":                        "TECHM.NS",
    "ntpc":                         "NTPC.NS",
    "power grid":                   "POWERGRID.NS",
    "power grid corporation":       "POWERGRID.NS",
    "ongc":                         "ONGC.NS",
    "oil and natural gas":          "ONGC.NS",
    "ultratech cement":             "ULTRACEMCO.NS",
    "ultratech":                    "ULTRACEMCO.NS",
    "nestle india":                 "NESTLEIND.NS",
    "nestle":                       "NESTLEIND.NS",
    "dr reddys":                    "DRREDDY.NS",
    "dr reddy's laboratories":      "DRREDDY.NS",
    "drreddy":                      "DRREDDY.NS",
    "tata motors":                  "TATAMOTORS.BO",
    "tatamotors":                   "TATAMOTORS.BO",
    "tata steel":                   "TATASTEEL.NS",
    "tatasteel":                    "TATASTEEL.NS",
    "jsw steel":                    "JSWSTEEL.NS",
    "jswsteel":                     "JSWSTEEL.NS",
    "hindalco":                     "HINDALCO.NS",
    "hindalco industries":          "HINDALCO.NS",
    "adani enterprises":            "ADANIENT.NS",
    "adani ports":                  "ADANIPORTS.NS",
    "adani green":                  "ADANIGREEN.NS",
    "adani total gas":              "ATGL.NS",
    "apollo hospitals":             "APOLLOHOSP.NS",
    "apollo":                       "APOLLOHOSP.NS",
    "cipla":                        "CIPLA.NS",
    "eicher motors":                "EICHERMOT.NS",
    "royal enfield":                "EICHERMOT.NS",
    "hero motocorp":                "HEROMOTOCO.NS",
    "hero":                         "HEROMOTOCO.NS",
    "bajaj auto":                   "BAJAJ-AUTO.NS",
    "bajaj":                        "BAJAJ-AUTO.NS",
    "divis laboratories":           "DIVISLAB.NS",
    "divi's":                       "DIVISLAB.NS",
    "divislab":                     "DIVISLAB.NS",
    "britannia":                    "BRITANNIA.NS",
    "britannia industries":         "BRITANNIA.NS",
    "grasim":                       "GRASIM.NS",
    "grasim industries":            "GRASIM.NS",
    "indusind bank":                "INDUSINDBK.NS",
    "indusindbk":                   "INDUSINDBK.NS",
    "shriram finance":              "SHRIRAMFIN.NS",
    "bpcl":                         "BPCL.NS",
    "bharat petroleum":             "BPCL.NS",
    "tata consumer":                "TATACONSUM.NS",
    "tata consumer products":       "TATACONSUM.NS",

    # ── Nifty Next 50 / Popular Mid-caps ──────────────────────
    "paytm":                        "PAYTM.NS",
    "one97 communications":         "PAYTM.NS",
    "nykaa":                        "NYKAA.NS",
    "fss":                          "NYKAA.NS",
    "policybazaar":                 "POLICYBZR.NS",
    "pg electroplast":              "PGEL.NS",
    "irctc":                        "IRCTC.NS",
    "indian railway catering":      "IRCTC.NS",
    "dmart":                        "DMART.NS",
    "avenue supermarts":            "DMART.NS",
    "pidilite":                     "PIDILITIND.NS",
    "pidilite industries":          "PIDILITIND.NS",
    "ltimindtree":                  "LTIM.NS",
    "lti mindtree":                 "LTIM.NS",
    "ltim":                         "LTIM.NS",
    "mphasis":                      "MPHASIS.NS",
    "persistent systems":           "PERSISTENT.NS",
    "persistent":                   "PERSISTENT.NS",
    "coforge":                      "COFORGE.NS",
    "info edge":                    "NAUKRI.NS",
    "naukri":                       "NAUKRI.NS",
    "indigo":                       "INDIGO.NS",
    "interglobe aviation":          "INDIGO.NS",
    "spicejet":                     "SPICEJET.NS",
    "vedanta":                      "VEDL.NS",
    "vedl":                         "VEDL.NS",
    "coal india":                   "COALINDIA.NS",
    "coalindia":                    "COALINDIA.NS",
    "hindustan zinc":               "HINDZINC.NS",
    "siemens":                      "SIEMENS.NS",
    "siemens india":                "SIEMENS.NS",
    "abb india":                    "ABB.NS",
    "abb":                          "ABB.NS",
    "havells":                      "HAVELLS.NS",
    "havells india":                "HAVELLS.NS",
    "voltas":                       "VOLTAS.NS",
    "blue dart":                    "BLUEDART.NS",
    "marico":                       "MARICO.NS",
    "dabur":                        "DABUR.NS",
    "godrej consumer":              "GODREJCP.NS",
    "godrej":                       "GODREJCP.NS",
    "colgate":                      "COLPAL.NS",
    "colgate palmolive":            "COLPAL.NS",
    "berger paints":                "BERGEPAINT.NS",
    "berger":                       "BERGEPAINT.NS",
    "kansai nerolac":               "KANSAINER.NS",
    "mrf":                          "MRF.NS",
    "ceat":                         "CEATLTD.NS",
    "apollo tyres":                 "APOLLOTYRE.NS",
    "motherson sumi":               "MOTHERSON.NS",
    "bosch":                        "BOSCHLTD.NS",
    "schaeffler india":             "SCHAEFFLER.NS",
    "srf":                          "SRF.NS",
    "pi industries":                "PIIND.NS",
    "upl":                          "UPL.NS",
    "united phosphorus":            "UPL.NS",
    "dalmia bharat":                "DALBHARAT.NS",
    "shree cement":                 "SHREECEM.NS",
    "ambuja cement":                "AMBUJACEM.NS",
    "acc":                          "ACC.NS",
    "varun beverages":              "VBL.NS",
    "united breweries":             "UBL.NS",
    "united spirits":               "MCDOWELL-N.NS",
    "radico khaitan":               "RADICO.NS",
    "trent":                        "TRENT.NS",
    "page industries":              "PAGEIND.NS",
    "jockey":                       "PAGEIND.NS",
    "max healthcare":               "MAXHEALTH.NS",
    "fortis healthcare":            "FORTIS.NS",
    "narayana hrudayalaya":         "NH.NS",
    "vijaya diagnostic":            "VIJAYA.NS",
    "syngene":                      "SYNGENE.NS",
    "ipca laboratories":            "IPCALAB.NS",
    "alkem laboratories":           "ALKEM.NS",
    "torrent pharma":               "TORNTPHARM.NS",
    "abbott india":                 "ABBOTINDIA.NS",
    "pfizer india":                 "PFIZER.NS",
    "glaxosmithkline":              "GLAXO.NS",
    "gsk pharma":                   "GLAXO.NS",
    "indraprastha gas":             "IGL.NS",
    "igl":                          "IGL.NS",
    "gujarat gas":                  "GUJGASLTD.NS",
    "petronet lng":                 "PETRONET.NS",
    "gail":                         "GAIL.NS",
    "gail india":                   "GAIL.NS",
    "indian oil":                   "IOC.NS",
    "ioc":                          "IOC.NS",
    "hindustan petroleum":          "HINDPETRO.NS",
    "hpcl":                         "HINDPETRO.NS",
    "bank of baroda":               "BANKBARODA.NS",
    "bankbaroda":                   "BANKBARODA.NS",
    "punjab national bank":         "PNB.NS",
    "pnb":                          "PNB.NS",
    "canara bank":                  "CANBK.NS",
    "union bank":                   "UNIONBANK.NS",
    "federal bank":                 "FEDERALBNK.NS",
    "idfc first bank":              "IDFCFIRSTB.NS",
    "idfc":                         "IDFCFIRSTB.NS",
    "au small finance bank":        "AUBANK.NS",
    "au bank":                      "AUBANK.NS",
    "bandhan bank":                 "BANDHANBNK.NS",
    "rbl bank":                     "RBLBANK.NS",
    "yes bank":                     "YESBANK.NS",
    "yesbank":                      "YESBANK.NS",
}

# ─────────────────────────────────────────────────────────────
# Ticker Resolver
# ─────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip extra spaces and punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"['\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_ticker(query: str) -> bool:
    """
    Return True only if the query looks like a deliberately typed ticker symbol.
    Rules:
      - Must be ALL CAPS (user typed it as a ticker, not a name)
      - OR ends with .NS / .BO explicitly
      - Lowercase words like "zomato" are treated as company names, not tickers
    """
    stripped = query.strip()
    # Explicit exchange suffix → always a ticker
    if stripped.upper().endswith(".NS") or stripped.upper().endswith(".BO"):
        return True
    # All uppercase, no spaces, reasonable length → ticker
    if stripped == stripped.upper() and stripped.isalpha() and 2 <= len(stripped) <= 20:
        return True
    # Contains digits mixed with letters (e.g. M&M, BAJAJ-AUTO) → ticker
    if stripped == stripped.upper() and bool(re.match(r"^[A-Z0-9\-&]{2,20}$", stripped)):
        return True
    return False


def resolve_ticker(query: str) -> tuple[str | None, str | None]:
    """
    Given a user query (name or ticker), return (ticker, display_name).
    Returns (None, None) if nothing is found.

    Strategy:
      1. Direct ticker input  → append .NS and verify with yfinance
      2. Exact local match    → return immediately
      3. Fuzzy local match    → suggest best match
      4. yfinance live search → fallback for unlisted stocks
    """
    query = query.strip()
    if not query:
        return None, None

    normalised = _normalise(query)

    # ── 1. Looks like a ticker ─────────────────────────────────
    if _is_ticker(query):
        ticker = query.upper()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        name = _verify_ticker(ticker)
        if name:
            return ticker, name
        # Try BSE as fallback
        bse_ticker = ticker.replace(".NS", ".BO")
        name = _verify_ticker(bse_ticker)
        if name:
            return bse_ticker, name
        return None, None

    # ── 2. Exact local match ───────────────────────────────────
    if normalised in NSE_STOCKS:
        ticker = NSE_STOCKS[normalised]
        name   = _verify_ticker(ticker) or query.title()
        return ticker, name

    # ── 3. Fuzzy local match ───────────────────────────────────
    close = get_close_matches(normalised, NSE_STOCKS.keys(), n=1, cutoff=0.6)
    if close:
        ticker = NSE_STOCKS[close[0]]
        name   = _verify_ticker(ticker) or close[0].title()
        return ticker, name

    # ── 4. Partial substring match ────────────────────────────
    for key, ticker in NSE_STOCKS.items():
        if normalised in key or key in normalised:
            name = _verify_ticker(ticker) or key.title()
            return ticker, name

    # ── 5. yfinance live search (last resort) ─────────────────
    return _yfinance_search(query)


def _verify_ticker(ticker: str) -> str | None:
    """
    Confirm the ticker exists on yfinance and return its long name.
    Uses fast_info only (1 call) to avoid slow .info fetches during search.
    Returns None if the ticker is invalid or unresolvable.
    """
    try:
        t          = yf.Ticker(ticker)
        fast       = t.fast_info
        last_price = fast.last_price
        if last_price and last_price > 0:
            # fast_info has a display name in newer yfinance versions
            name = getattr(fast, "display_name", None)
            if not name:
                # Only fall back to .info if fast_info doesn't have a name
                try:
                    info = t.info
                    name = info.get("longName") or info.get("shortName")
                except Exception:
                    name = None
            return name or ticker
    except Exception:
        pass
    return None


def _yfinance_search(query: str) -> tuple[str | None, str | None]:
    """
    Use yfinance's search to find an NSE ticker for a company name.
    """
    try:
        results = yf.Search(query, max_results=10).quotes
        # Prefer NSE (.NS) results
        for r in results:
            sym = r.get("symbol", "")
            if sym.endswith(".NS"):
                name = r.get("longname") or r.get("shortname") or sym
                return sym, name
        # Fallback to first result
        if results:
            sym  = results[0].get("symbol", "")
            name = results[0].get("longname") or results[0].get("shortname") or sym
            if sym:
                return sym, name
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────
# Search Suggestions  (for autocomplete-style display)
# ─────────────────────────────────────────────────────────────

def get_suggestions(query: str, max_results: int = 6) -> list[dict]:
    """
    Return up to `max_results` matching stocks from the local database
    for displaying as live suggestions while the user types.
    Deduplicates by ticker so the same stock never appears twice.
    """
    if not query or len(query) < 2:
        return []

    normalised     = _normalise(query)
    seen_tickers   = set()
    matches        = []

    # Substring matches first — prefer the "nicest" name for each ticker
    for name, ticker in NSE_STOCKS.items():
        if normalised in name and ticker not in seen_tickers:
            # Use the longest name for this ticker as the display label
            display = max(
                (n for n, t in NSE_STOCKS.items() if t == ticker),
                key=len
            ).title()
            matches.append({"name": display, "ticker": ticker})
            seen_tickers.add(ticker)
        if len(matches) >= max_results:
            break

    # Fuzzy fill if still under max_results
    if len(matches) < max_results:
        close = get_close_matches(
            normalised, NSE_STOCKS.keys(),
            n=max_results * 2,   # fetch extra to account for dedup
            cutoff=0.5
        )
        for c in close:
            t = NSE_STOCKS[c]
            if t not in seen_tickers:
                display = max(
                    (n for n, tk in NSE_STOCKS.items() if tk == t),
                    key=len
                ).title()
                matches.append({"name": display, "ticker": t})
                seen_tickers.add(t)
            if len(matches) >= max_results:
                break

    return matches[:max_results]