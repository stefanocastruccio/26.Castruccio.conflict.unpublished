"""Download all ACLED events, worldwide, year by year (1997-2025), through the ACLED API (OAuth).

Credentials are read from ~/.acled_credentials (lines ACLED_EMAIL=..., ACLED_PASSWORD=...;
chmod 600). They are never printed, logged or written anywhere else. Raw events are ACLED data
and must stay outside the repository (ACLED terms of use): write them to $WORK only.

Output: <out>/acled_<year>.parquet (all fields as returned, strings), plus <out>/manifest.json
with row counts per year. Years already present are skipped (resume after interruption).

Usage: python download_acled.py --out $WORK/acled/raw [--years 1997 2025] [--limit 5000]

API (check against https://acleddata.com/api-documentation/ before the first full run):
  token:  POST https://acleddata.com/oauth/token  form: username, password, grant_type=password,
          client_id=acled  -> access_token (about 24 h), refresh_token
  read:   GET  https://acleddata.com/api/acled/read?_format=json  header Authorization: Bearer <token>
          params: year, limit, page  -> {"success", "count", "data": [...]}
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time

import pandas as pd
import requests

BASE = os.environ.get("ACLED_BASE", "https://acleddata.com")
TOKEN_URL = f"{BASE}/oauth/token"
READ_URL = f"{BASE}/api/acled/read"


def credentials(path=os.path.expanduser("~/.acled_credentials")):
    st = os.stat(path)
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        sys.exit(f"{path} is readable by others; run chmod 600 {path}")
    kv = {}
    for line in open(path):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    if not kv.get("ACLED_EMAIL") or not kv.get("ACLED_PASSWORD"):
        sys.exit(f"{path} must define ACLED_EMAIL and ACLED_PASSWORD")
    return kv["ACLED_EMAIL"], kv["ACLED_PASSWORD"]


class Session:
    def __init__(self):
        self.email, self._pw = credentials()
        self.http = requests.Session()
        self.token = self.refresh = None
        self.login()

    def login(self):
        r = self.http.post(TOKEN_URL, data={"username": self.email, "password": self._pw,
                                            "grant_type": "password", "client_id": "acled"}, timeout=60)
        if r.status_code != 200:
            sys.exit(f"token request failed: HTTP {r.status_code} {r.text[:300]}")
        j = r.json()
        self.token, self.refresh = j["access_token"], j.get("refresh_token")
        print("token obtained", flush=True)

    def renew(self):
        if self.refresh:
            r = self.http.post(TOKEN_URL, data={"refresh_token": self.refresh, "grant_type": "refresh_token",
                                                "client_id": "acled"}, timeout=60)
            if r.status_code == 200:
                j = r.json()
                self.token, self.refresh = j["access_token"], j.get("refresh_token", self.refresh)
                print("token refreshed", flush=True)
                return
        self.login()

    def get(self, params, tries=6):
        for k in range(tries):
            try:
                r = self.http.get(READ_URL, params=dict(params, _format="json"),
                                  headers={"Authorization": f"Bearer {self.token}"}, timeout=300)
            except requests.RequestException as e:
                print(f"  network error ({e.__class__.__name__}); retry {k + 1}", flush=True)
                time.sleep(2 ** (k + 1)); continue
            if r.status_code == 401:
                self.renew(); continue
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"  HTTP {r.status_code}; retry {k + 1}", flush=True)
                time.sleep(2 ** (k + 2)); continue
            if r.status_code != 200:
                sys.exit(f"read failed: HTTP {r.status_code} {r.text[:300]}")
            j = r.json()
            if j.get("success") is False:
                sys.exit(f"read failed: {json.dumps(j)[:300]}")
            return j
        sys.exit("read failed after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[1997, 2025])
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--test", action="store_true", help="one page of 1997 only, nothing written")
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))):
        sys.exit("--out is inside the repository; raw ACLED data must stay in $WORK")
    os.makedirs(a.out, exist_ok=True)
    s = Session()
    if a.test:
        j = s.get({"year": a.years[0], "limit": 5, "page": 1})
        print("test ok:", j.get("count"), "rows; fields:", sorted(j["data"][0]) if j.get("data") else None)
        return
    man_path = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for year in range(a.years[0], a.years[1] + 1):
        f = os.path.join(a.out, f"acled_{year}.parquet")
        if os.path.exists(f):
            print(f"{year}: present, skipped", flush=True); continue
        rows, page, t0 = [], 1, time.time()
        while True:
            j = s.get({"year": year, "limit": a.limit, "page": page})
            data = j.get("data", [])
            rows += data
            if len(data) < a.limit:
                break
            page += 1
            if page % 20 == 0:
                print(f"  {year}: page {page}, {len(rows):,} rows", flush=True)
        df = pd.DataFrame(rows).astype("string")
        if len(df) and "event_id_cnty" in df:
            n0 = len(df); df = df.drop_duplicates("event_id_cnty")
            if len(df) < n0:
                print(f"  {year}: dropped {n0 - len(df)} duplicate event ids (pagination overlap)", flush=True)
        df.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)
        man[str(year)] = {"rows": int(len(df)), "pages": page, "seconds": round(time.time() - t0, 1),
                          "downloaded": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(man, open(man_path, "w"), indent=1)
        print(f"{year}: {len(df):,} events in {page} pages ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
