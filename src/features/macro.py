"""Macroeconomic features from CSV and optional CBR API."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests

CBR_SOAP_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
CBR_DAILY_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


def _add_lag_diff_features(frame: pd.DataFrame, col: str, diff_lag: int = 5) -> pd.DataFrame:
    out = frame.copy()
    out[f"{col}_lag1"] = out[col].shift(1)
    out[f"{col}_diff{diff_lag}"] = out[col].diff(diff_lag)
    return out


def build_macro_features_from_csv(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build macro features from columns already present in data.csv."""
    macro_cfg = config.get("macro", {})
    columns = macro_cfg.get(
        "csv_columns",
        ["IMICEX", "TransRUB1M", "covid", "IsDayOff_Status_Workalendar_RU"],
    )
    diff_lag = macro_cfg.get("macro_diff_lag", 5)

    out = pd.DataFrame({"Date": pd.to_datetime(df["Date"])})
    work = df.set_index("Date")

    for col in columns:
        if col not in work.columns:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        tmp = pd.DataFrame({"Date": work.index, col: series.values})
        tmp = _add_lag_diff_features(tmp, col, diff_lag=diff_lag)
        out = out.merge(tmp.drop(columns=[col]), on="Date", how="left")

    return out


def load_cbr_key_rate(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch CBR key rate history via SOAP API."""
    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                   xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <KeyRate xmlns="http://web.cbr.ru/">
          <fromDate>{date_from}</fromDate>
          <ToDate>{date_to}</ToDate>
        </KeyRate>
      </soap:Body>
    </soap:Envelope>"""
    headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": "http://web.cbr.ru/KeyRate"}
    response = requests.post(CBR_SOAP_URL, data=envelope.encode("utf-8"), headers=headers, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows = []
    for kr in root.iter():
        if kr.tag.endswith("KR"):
            date_val = next((c.text for c in kr if c.tag.endswith("DT")), None)
            rate_val = next((c.text for c in kr if c.tag.endswith("Rate")), None)
            if date_val and rate_val:
                rows.append({"Date": pd.to_datetime(date_val), "key_rate": float(rate_val.replace(",", "."))})
    return pd.DataFrame(rows)


def load_ruonia(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch RUONIA rates via CBR SOAP API."""
    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                   xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <Ruonia xmlns="http://web.cbr.ru/">
          <fromDate>{date_from}</fromDate>
          <ToDate>{date_to}</ToDate>
        </Ruonia>
      </soap:Body>
    </soap:Envelope>"""
    headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": "http://web.cbr.ru/Ruonia"}
    response = requests.post(CBR_SOAP_URL, data=envelope.encode("utf-8"), headers=headers, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows = []
    for item in root.iter():
        if item.tag.endswith("ro"):
            date_val = next((c.text for c in item if c.tag.endswith("D0")), None)
            rate_val = next((c.text for c in item if c.tag.endswith("ruo")), None)
            if date_val and rate_val:
                rows.append({"Date": pd.to_datetime(date_val), "ruonia": float(rate_val.replace(",", "."))})
    return pd.DataFrame(rows)


def load_usd_rub(date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch USD/RUB daily rates from CBR XML endpoint."""
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    rows = []
    for day in pd.date_range(start, end, freq="D"):
        params = {"date_req": day.strftime("%d/%m/%Y")}
        response = requests.get(CBR_DAILY_URL, params=params, timeout=30)
        if response.status_code != 200:
            continue
        root = ET.fromstring(response.content)
        for valute in root.findall("Valute"):
            char_code = valute.findtext("CharCode")
            if char_code != "USD":
                continue
            nominal = float(valute.findtext("Nominal", "1"))
            value = float(valute.findtext("Value", "0").replace(",", "."))
            rows.append({"Date": day, "usd_rub": value / nominal})
    return pd.DataFrame(rows)


def build_macro_features_from_api(dates: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    """Fetch macro series from CBR and align to daily calendar."""
    macro_cfg = config.get("macro", {})
    date_from = macro_cfg.get("api_start_date", str(dates.min().date()))
    date_to = macro_cfg.get("api_end_date", str(dates.max().date()))
    diff_lag = macro_cfg.get("macro_diff_lag", 5)

    calendar = pd.DataFrame({"Date": pd.to_datetime(dates).sort_values().unique()})
    key_rate = load_cbr_key_rate(date_from, date_to)
    ruonia = load_ruonia(date_from, date_to)
    usd = load_usd_rub(date_from, date_to)

    out = calendar.merge(key_rate, on="Date", how="left")
    out = out.merge(ruonia, on="Date", how="left")
    out = out.merge(usd, on="Date", how="left")
    out = out.sort_values("Date").ffill().bfill()
    out["ruonia_minus_key"] = out["ruonia"] - out["key_rate"]

    for col in ["key_rate", "ruonia", "usd_rub", "ruonia_minus_key"]:
        if col in out.columns:
            out = _add_lag_diff_features(out, col, diff_lag=diff_lag)

    return out


def build_macro_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build macro features using CSV by default and optional CBR API enrichment."""
    macro_cfg = config.get("macro", {})
    csv_features = build_macro_features_from_csv(df, config)

    if not macro_cfg.get("use_cbr_api", False):
        return csv_features

    api_features = build_macro_features_from_api(df["Date"], config)
    merged = csv_features.merge(api_features, on="Date", how="left", suffixes=("", "_api"))
    return merged
