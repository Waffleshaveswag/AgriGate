"""Area-weighted surface-soil averages for every SSURGO survey area (≈ county) in a state, via USDA Soil Data Access."""
import json, sys, time, urllib.request
SDA = "https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest"
PROPS = {"ch.ph1to1h2o_r": "ph", "ch.om_r": "om_pct", "ch.cec7_r": "cec", "ch.awc_r": "awc", "ch.sandtotal_r": "sand_pct",
         "ch.silttotal_r": "silt_pct", "ch.claytotal_r": "clay_pct", "ch.caco3_r": "caco3_pct", "ch.ec_r": "ec_dsm",
         "ch.kwfact": "k_factor", "c.slope_r": "slope_pct"}

def state_query(st):
    w = "CAST(c.comppct_r AS FLOAT) * mu.muacres"
    sel = ",\n".join(f"SUM(CASE WHEN {k} IS NOT NULL THEN {k} * {w} END) / NULLIF(SUM(CASE WHEN {k} IS NOT NULL THEN {w} END), 0) AS {v}"
                     for k, v in PROPS.items())
    poor = "SUM(CASE WHEN c.drainagecl IN ('Poorly drained','Very poorly drained') THEN " + w + " ELSE 0 END) / NULLIF(SUM(" + w + "), 0) AS poorly_drained_share"
    sql = f"""SELECT l.areasymbol, {sel}, {poor}, SUM(mu.muacres) AS acres
      FROM legend l JOIN mapunit mu ON mu.lkey = l.lkey
      JOIN component c ON c.mukey = mu.mukey AND c.majcompflag = 'Yes'
      JOIN chorizon ch ON ch.cokey = c.cokey AND ch.hzdept_r = 0
      WHERE l.areasymbol LIKE '{st}%' AND mu.muacres > 0 GROUP BY l.areasymbol"""
    body = json.dumps({"format": "JSON+COLUMNNAME", "query": sql}).encode()
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(SDA, data=body, headers={"Content-Type": "application/json"}), timeout=300) as r:
                t = json.load(r).get("Table", [])
                return [dict(zip(t[0], row)) for row in t[1:]] if t else []
        except Exception as e:
            err = e; time.sleep(5 * (k + 1))
    raise RuntimeError(f"{st}: {err}")

if __name__ == "__main__":
    rows = state_query(sys.argv[1])
    print(len(rows)); print(rows[:2])
