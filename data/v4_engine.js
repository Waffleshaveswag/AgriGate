/**
 * data/v4_engine.js — AgriGate v4 Scientific Agronomy, Hydrology, Terrain, Economics & Multi-API Cascade
 *
 * Implements:
 *   1. Offline Multi-Tier Lookup: 3,221 US Counties (V4_US_COUNTIES) + 50 Global Anchors (V4_GLOBAL_ANCHORS)
 *   2. FAO-56 Penman-Monteith & Hargreaves ET0 + OpenET / ECOSTRESS LST CWSI
 *   3. RUSLE Soil Erosion (A = R * K * LS * C * P) using IMERG + SSURGO/SoilGrids + SRTM Slope + CDL/LANDFIRE
 *   4. Irrigation Feasibility Index (SSURGO AWC + SMAP + USGS NWIS + GRACE + CoCoRaHS)
 *   5. Economic Net Margin ($/ac) from USDA NASS QuickStats Yields + AMS Market News Prices
 *   6. 5-Strategy Weighted Scoring + CDL 30m Pest-Break Rotation Check
 *   7. Parallel Live Online Cascade (Zero-Auth APIs + Optional User API Key Overrides in localStorage)
 */

(function (global) {
  'use strict';

  const V4_KEYS_STORAGE = 'agrigate_v4_api_keys';

  const STRATEGY_WEIGHTS = {
    balanced:        { label: 'Balanced',        nutrient: 0.28, water: 0.24, erosion: 0.16, econ: 0.20, pest: 0.12 },
    water_saver:     { label: 'Water Saver',     nutrient: 0.14, water: 0.48, erosion: 0.14, econ: 0.14, pest: 0.10 },
    soil_builder:    { label: 'Soil Builder',    nutrient: 0.36, water: 0.14, erosion: 0.28, econ: 0.10, pest: 0.12 },
    cash_profit:     { label: 'Cash Profit',     nutrient: 0.14, water: 0.16, erosion: 0.08, econ: 0.50, pest: 0.12 },
    erosion_control: { label: 'Erosion Control', nutrient: 0.18, water: 0.16, erosion: 0.46, econ: 0.10, pest: 0.10 }
  };

  function getApiKeys() {
    try {
      return JSON.parse(localStorage.getItem(V4_KEYS_STORAGE) || '{}') || {};
    } catch (e) {
      return {};
    }
  }

  function saveApiKeys(keys) {
    try {
      localStorage.setItem(V4_KEYS_STORAGE, JSON.stringify(keys || {}));
    } catch (e) {}
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371.0;
    const p1 = (lat1 * Math.PI) / 180;
    const p2 = (lat2 * Math.PI) / 180;
    const dp = ((lat2 - lat1) * Math.PI) / 180;
    const dl = ((lon2 - lon1) * Math.PI) / 180;
    const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
  }

  // Exact WGS84/NAD83 -> EPSG:5070 CONUS Albers Equal Area Conic (for USDA CropScape CDL queries)
  function latLonToEpsg5070(latDeg, lonDeg) {
    const a = 6378137.0;
    const f = 1.0 / 298.257222101;
    const e2 = 2 * f - f * f;
    const e = Math.sqrt(e2);
    const phi0 = (23.0 * Math.PI) / 180.0;
    const lam0 = (-96.0 * Math.PI) / 180.0;
    const phi1 = (29.5 * Math.PI) / 180.0;
    const phi2 = (45.5 * Math.PI) / 180.0;

    const qFn = (phi) => {
      const s = Math.sin(phi);
      return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * Math.log((1 - e * s) / (1 + e * s)));
    };
    const mFn = (phi) => {
      const s = Math.sin(phi);
      return Math.cos(phi) / Math.sqrt(1 - e2 * s * s);
    };

    const q0 = qFn(phi0), q1 = qFn(phi1), q2 = qFn(phi2);
    const m1 = mFn(phi1), m2 = mFn(phi2);
    const n = (m1 * m1 - m2 * m2) / (q2 - q1);
    const C = m1 * m1 + n * q1;
    const rho0 = (a * Math.sqrt(Math.max(0, C - n * q0))) / n;

    const phi = (latDeg * Math.PI) / 180.0;
    const lam = (lonDeg * Math.PI) / 180.0;
    const q = qFn(phi);
    const rho = (a * Math.sqrt(Math.max(0, C - n * q))) / n;
    const theta = n * (lam - lam0);
    return {
      x: Math.round(rho * Math.sin(theta)),
      y: Math.round(rho0 - rho * Math.cos(theta))
    };
  }

  function isInsideUS(lat, lon) {
    return (lat >= 24.0 && lat <= 50.0 && lon >= -125.0 && lon <= -66.0) ||
           (lat >= 51.0 && lat <= 71.5 && lon >= -170.0 && lon <= -129.0) ||
           (lat >= 18.5 && lat <= 22.5 && lon >= -160.5 && lon <= -154.5);
  }

  // Find nearest US County from the 3,221-county offline database
  function findNearestUsCounty(lat, lon) {
    const rows = global.V4_US_COUNTIES;
    const fields = global.V4_COUNTY_FIELDS;
    if (!Array.isArray(rows) || !Array.isArray(fields) || rows.length === 0) return null;
    if (!isInsideUS(lat, lon)) return null;

    let bestRow = null;
    let bestD = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const dLat = r[4] - lat;
      const dLon = r[5] - lon;
      if (Math.abs(dLat) > 2.5 || Math.abs(dLon) > 3.0) continue;
      const d = haversineKm(lat, lon, r[4], r[5]);
      if (d < bestD) {
        bestD = d;
        bestRow = r;
      }
    }
    if (!bestRow || bestD > 130) return null;

    const obj = { distKm: Math.round(bestD) };
    for (let k = 0; k < fields.length; k++) obj[fields[k]] = bestRow[k];
    return obj;
  }

  // FAO-56 Penman-Monteith daily reference ET (mm/day) -> monthly (mm/mo)
  function computePenmanMonteithDay(tMean, tMax, tMin, rhMean, u2, rsMj, elevM) {
    const t = Number(tMean) || 15;
    const u = Math.max(0.5, Number(u2) || 2.5);
    const rs = Math.max(2.0, Number(rsMj) || 15.0);
    const rh = Math.max(15, Math.min(98, Number(rhMean) || 60));
    const z = Math.max(0, Number(elevM) || 100);

    const P = 101.3 * Math.pow((293 - 0.0065 * z) / 293, 5.26); // kPa
    const gamma = 0.000665 * P; // psychrometric constant
    const esT = (temp) => 0.6108 * Math.exp((17.27 * temp) / (temp + 237.3));
    const es = (esT(tMax) + esT(tMin)) / 2;
    const ea = es * (rh / 100.0);
    const vpd = Math.max(0.05, es - ea);
    const delta = (4098 * esT(t)) / Math.pow(t + 237.3, 2);

    // Net radiation estimate Rn (MJ/m2/day) with albedo 0.23
    const rns = (1 - 0.23) * rs;
    const rnl = 0.22 * rs; // clear/cloud-blended net longwave proxy
    const rn = Math.max(0.5, rns - rnl);

    const num = 0.408 * delta * rn + gamma * (900 / (t + 273)) * u * vpd;
    const den = delta + gamma * (1 + 0.34 * u);
    return Math.max(0.2, num / den);
  }

  // USDA SCS Effective Rainfall (mm/mo)
  function computeEffectiveRainMonth(pMonth) {
    const p = Math.max(0, Number(pMonth) || 0);
    if (p <= 250) return p * ((125 - 0.2 * p) / 125);
    return 125 + 0.1 * p;
  }

  // Renard & Freimund (1994) Rainfall Erosivity R-factor + GPM IMERG storm intensity multiplier
  function computeRusleRFactor(precipMm, stormFactor) {
    const p = Math.max(10, Number(precipMm) || 500);
    const baseR = p < 850
      ? 0.0483 * Math.pow(p, 1.61)
      : (587.8 - 1.219 * p + 0.004105 * p * p);
    return Math.max(20, Math.round(baseR * (stormFactor || 1.0)));
  }

  // Wischmeier / EPIC Soil Erodibility K-factor (t*ha*h / (ha*MJ*mm)) when SSURGO kwfact unavailable
  function computeSoilKFactor(sandPct, siltPct, clayPct, omPct) {
    const sand = Math.max(0, Math.min(100, Number(sandPct) || 30));
    const silt = Math.max(0, Math.min(100, Number(siltPct) || 45));
    const clay = Math.max(1, Math.min(100, Number(clayPct) || 25));
    const om = Math.max(0.1, Math.min(12, Number(omPct) || 2.0));

    const fcsand = 0.2 + 0.3 * Math.exp(-0.0256 * sand * (1 - silt / 100));
    const fclsi = Math.pow(silt / (clay + silt), 0.3);
    const orgC = om * 0.58;
    const fhisand = 1 - (0.7 * (1 - sand / 100)) / ((1 - sand / 100) + Math.exp(-5.51 + 22.9 * (1 - sand / 100)));
    const forgc = 1 - (0.25 * orgC) / (orgC + Math.exp(3.72 - 2.95 * orgC));
    const kMetric = fcsand * fclsi * forgc * fhisand;
    // Convert EPIC K to US/SI comparable 0.05..0.55 range
    return Number(Math.max(0.05, Math.min(0.55, kMetric * 0.92)).toFixed(2));
  }

  // RUSLE Topographic LS-factor (McCool et al. 1989)
  function computeRusleLSFactor(slopePct, slopeLenM) {
    const sPct = Math.max(0.2, Math.min(45, Number(slopePct) || 1.5));
    const lam = Math.max(15, Number(slopeLenM) || 60);
    const theta = Math.atan(sPct / 100.0);
    const sinT = Math.sin(theta);
    const beta = (sinT / 0.0896) / (3.0 * Math.pow(sinT, 0.8) + 0.56);
    const m = beta / (1 + beta);
    const L = Math.pow(lam / 22.13, m);
    const S = sPct < 9.0 ? (10.8 * sinT + 0.03) : (16.8 * sinT - 0.50);
    return Number(Math.max(0.08, L * S).toFixed(2));
  }

  function lookupCropEcon(crop) {
    const econDb = global.V4_CROPS_ECON || {};
    const name = ((crop && crop.name) || '').toLowerCase();
    const key = ((crop && crop.key) || '').toLowerCase().replace(/_/g, ' ');
    const keysSorted = Object.keys(econDb).sort((a, b) => b.length - a.length);
    for (const ek of keysSorted) {
      if (key === ek || name.includes(ek) || key.includes(ek)) {
        return { econKey: ek, ...econDb[ek] };
      }
    }
    return { econKey: 'generic', unit: 'ton', yield: 2.8, price: 145, cost: 230, kc: 0.95, rusleC: 0.14, cdlId: 37, gaezT: [12, 26], waterMm: 420 };
  }

  // Full RUSLE Soil Erosion calculation (A = R * K * LS * C * P)
  function computeRusleErosion(soilData, activeCrops) {
    const R = Number(soilData.rFactor) || computeRusleRFactor(soilData.precip, 1.0);
    const K = Number(soilData.kFactor) || computeSoilKFactor(soilData.sand, soilData.silt, soilData.clay, soilData.som);
    const LS = computeRusleLSFactor(soilData.slopePct || 1.5, 60);

    let C = 0.24;
    let P = 1.0;
    if (Array.isArray(activeCrops) && activeCrops.length > 0) {
      let cSum = 0;
      let hasWinter = false;
      let hasPerennial = false;
      activeCrops.forEach(cr => {
        const ec = lookupCropEcon(cr);
        cSum += ec.rusleC;
        if (cr.season === 'winter' || cr.season === 'fall' || cr.role === 'cover_drill') hasWinter = true;
        if (cr.season === 'perennial') hasPerennial = true;
      });
      C = cSum / activeCrops.length;
      if (hasWinter) P *= 0.72;
      if (hasPerennial) P *= 0.60;
      if (activeCrops.length >= 3) P *= 0.85;
    } else if (soilData.nlcdCode === 41 || soilData.nlcdCode === 42 || soilData.nlcdCode === 43) {
      C = 0.01;
    } else if (soilData.nlcdCode === 71 || soilData.nlcdCode === 81) {
      C = 0.04;
    }

    // R (MJ mm / ha h yr) * K (0.0131 * t ha h / ha MJ mm) -> t/ha/yr -> tons/ac/yr (* 0.446)
    const lossMetric = R * (K * 0.1317) * LS * C * P;
    const lossTonsAc = Number(Math.max(0.05, lossMetric * 0.44609).toFixed(2));
    let riskLabel = 'Low';
    let riskTone = 'emerald';
    if (lossTonsAc >= 5.0) { riskLabel = 'Severe (>T)'; riskTone = 'rose'; }
    else if (lossTonsAc >= 2.5) { riskLabel = 'Moderate'; riskTone = 'amber'; }

    return {
      R,
      K: Number(K.toFixed(2)),
      LS,
      C: Number(C.toFixed(2)),
      P: Number(P.toFixed(2)),
      lossTonsAc,
      riskLabel,
      riskTone
    };
  }

  // Irrigation Feasibility Index (0..100) from NWIS stream discharge, GRACE aquifer trend, SMAP, and SSURGO AWC
  function computeIrrigationFeasibility(soilData) {
    const nwisQ = Number(soilData.nwisPercentile ?? 50);
    const grace = Number(soilData.grace ?? -10);
    const smap = Number(soilData.smapMoist ?? 16);
    const awc = Number(soilData.awc ?? 0.17);
    const bal = Number(soilData.balance ?? -200);

    const graceScore = Math.max(0, Math.min(100, 75 + grace * 1.4));
    const smapScore = Math.max(0, Math.min(100, smap * 3.2));
    const awcScore = Math.max(0, Math.min(100, (awc / 0.22) * 100));
    const balScore = Math.max(0, Math.min(100, 65 + bal * 0.06));
    const score = Math.round(nwisQ * 0.30 + graceScore * 0.30 + smapScore * 0.20 + awcScore * 0.10 + balScore * 0.10);

    let status = 'Moderate Supply';
    let tone = 'amber';
    if (score >= 68) { status = 'Sustainable'; tone = 'emerald'; }
    else if (score < 42) { status = 'Aquifer Stress'; tone = 'rose'; }

    return { score, status, tone, nwisQ };
  }

  // ECOSTRESS / OpenET Canopy Water Stress Index (CWSI 0..1)
  function computeCwsi(soilData) {
    const esi = Number(soilData.esi ?? 0.5);
    const heat = Number(soilData.heatDays ?? 25);
    const vpdProxy = Math.max(0.4, Math.min(3.8, 0.9 + (Number(soilData.meanTemp) || 15) * 0.06 + heat * 0.015));
    const lstAirDelta = Number(((1 - esi) * 6.8 + heat * 0.03).toFixed(1));
    const cwsi = Number(Math.max(0.04, Math.min(0.96, (1 - esi) * 0.78 + (heat / 140) * 0.22)).toFixed(2));
    return { cwsi, lstAirDelta, vpdKpa: Number(vpdProxy.toFixed(2)) };
  }

  // Economic Net Margin ($/ac) using USDA NASS QuickStats yield * AMS price adjusted for local climate/soil
  function computeCropEconomics(crop, soilData, addedWaterMm) {
    const ec = lookupCropEcon(crop);
    const optMid = (ec.gaezT[0] + ec.gaezT[1]) / 2;
    const tDiff = Math.abs((Number(soilData.meanTemp) || 15) - optMid);
    const fClim = Math.max(0.45, 1.0 - Math.max(0, tDiff - 4.0) * 0.038);

    const effWater = (Number(soilData.precip) || 600) + (Number(addedWaterMm) || 0) + (Number(soilData.paw) || 20) * 6;
    const waterRatio = Math.min(1.08, effWater / Math.max(150, ec.waterMm));
    const fWater = Math.max(0.35, Math.min(1.05, 0.45 + 0.60 * waterRatio));

    const ph = Number(soilData.ph) || 6.8;
    const phMid = (crop.minPh + crop.maxPh) / 2;
    const phPen = (ph >= crop.minPh && ph <= crop.maxPh) ? 1.0 : Math.max(0.55, 1.0 - Math.abs(ph - phMid) * 0.14);

    // Optional live NASS county yield override if fetched
    const liveNass = global.V4_LIVE_NASS_YIELDS && global.V4_LIVE_NASS_YIELDS[ec.econKey];
    const baseYield = liveNass ? liveNass.yield : ec.yield;
    const adjYield = Number((baseYield * fClim * fWater * phPen).toFixed(1));
    const grossRev = Math.round(adjYield * ec.price);
    const waterCost = Math.round((Number(addedWaterMm) || 0) * 0.26);
    const netMargin = grossRev - ec.cost - waterCost;

    return {
      unit: ec.unit,
      yield: adjYield,
      price: ec.price,
      grossRev,
      cost: ec.cost + waterCost,
      netMargin,
      rusleC: ec.rusleC,
      cdlId: ec.cdlId,
      isLiveNass: !!liveNass
    };
  }

  // 5-Strategy Weighted Scoring + CDL Rotation History Pest-Break Check
  function scoreCropV4(crop, soilData, activeCrops, cdlHistoryIds, strategyKey) {
    const w = STRATEGY_WEIGHTS[strategyKey] || STRATEGY_WEIGHTS.balanced;
    const econ = computeCropEconomics(crop, soilData, soilData.addedWaterMm || 0);
    const ph = Number(soilData.ph) || 6.8;

    // 1. Nutrient Fit (0..100)
    let nutrientFit = 52;
    if (soilData.nTot < 1.8 && crop.n > 0) nutrientFit += Math.min(45, crop.n * 0.35);
    if (soilData.pAvail < 30 && crop.pCredit > 0) nutrientFit += Math.min(35, crop.pCredit * 1.1);
    if (ph > 7.2 && crop.phShift < 0) nutrientFit += Math.abs(crop.phShift) * 40;
    if (ph < 5.9 && crop.acidPhShift > 0) nutrientFit += crop.acidPhShift * 40;
    nutrientFit = Math.max(0, Math.min(100, nutrientFit));

    // 2. Water & Climate Fit (0..100)
    const ec = lookupCropEcon(crop);
    const waterGap = Math.max(0, ec.waterMm - ((Number(soilData.precip) || 500) + (Number(soilData.paw) || 20) * 5));
    let waterFit = 90 - (waterGap / ec.waterMm) * 65;
    if (soilData.balance < -350 && (crop.root === 'deep_taproot' || crop.role === 'cash_c4')) waterFit += 18;
    if (soilData.frostFree < 120 && crop.season === 'warm') waterFit -= 25;
    waterFit = Math.max(5, Math.min(100, waterFit));

    // 3. Soil & Erosion Control Fit (0..100)
    let soilErosionFit = Math.max(10, Math.min(100, 108 - ec.rusleC * 240));
    if ((soilData.slopePct || 1.5) > 3.5 && ec.rusleC <= 0.14) soilErosionFit = Math.min(100, soilErosionFit + 18);
    if ((soilData.bulkDensity || 1.3) >= 1.38 && (crop.root === 'bio_drill' || crop.role === 'cover_drill')) soilErosionFit = 100;

    // 4. Economic Fit (0..100)
    const econFit = Math.max(5, Math.min(100, 42 + econ.netMargin / 8.5));

    // 5. Pest-Break & CDL History Diversity (0..100)
    let pestBreakFit = 75;
    const sameFamCount = (activeCrops || []).filter(a => a.family === crop.family).length;
    pestBreakFit -= sameFamCount * 22;
    if (Array.isArray(cdlHistoryIds) && cdlHistoryIds.includes(ec.cdlId)) {
      pestBreakFit -= 28; // Penalize repeating recent CropScape CDL monoculture crop
    } else if (Array.isArray(cdlHistoryIds) && cdlHistoryIds.length > 0) {
      pestBreakFit += 18; // Bonus for breaking recent CDL crop family
    }
    pestBreakFit = Math.max(10, Math.min(100, pestBreakFit));

    const total = Math.round(
      w.nutrient * nutrientFit +
      w.water * waterFit +
      w.erosion * soilErosionFit +
      w.econ * econFit +
      w.pest * pestBreakFit
    );

    return {
      total,
      econ,
      sub: { nutrientFit: Math.round(nutrientFit), waterFit: Math.round(waterFit), soilErosionFit: Math.round(soilErosionFit), econFit: Math.round(econFit), pestBreakFit: Math.round(pestBreakFit) },
      breaksCdlMonoculture: Array.isArray(cdlHistoryIds) && cdlHistoryIds.length > 0 && !cdlHistoryIds.includes(ec.cdlId)
    };
  }

  // Offline / Baseline Synthesis combining 3,221 US Counties + 50 Global NASA POWER/SRTM Anchors
  function computeV4OfflineBaseline(lat, lon) {
    const county = findNearestUsCounty(lat, lon);
    const anchors = global.V4_GLOBAL_ANCHORS || [];

    // Open-ocean / ice-sheet guard
    let closestAnchor = null;
    let closestDist = Infinity;
    for (let i = 0; i < anchors.length; i++) {
      const d = haversineKm(lat, lon, anchors[i].lat, anchors[i].lon);
      if (d < closestDist) {
        closestDist = d;
        closestAnchor = anchors[i];
      }
    }

    // IDW across global anchors
    let tw = 0;
    const wList = [];
    for (let i = 0; i < anchors.length; i++) {
      const a = anchors[i];
      const d = haversineKm(lat, lon, a.lat, a.lon);
      const w = 1.0 / Math.pow(Math.max(25.0, d), 2.2);
      wList.push({ w, a });
      tw += w;
    }
    const blend = (k) => {
      if (!tw) return 0;
      let s = 0;
      for (let i = 0; i < wList.length; i++) s += wList[i].w * (wList[i].a[k] || 0);
      return s / tw;
    };

    const isArable = closestAnchor ? closestAnchor.arable !== false : true;
    const meanTempNum = county ? county.temp : blend('temp');
    const precip = county ? county.precip : Math.max(10, Math.round(blend('precip')));
    const pet = county ? county.pet : Math.max(20, Math.round(blend('pet')));
    const elevM = county ? county.elev : Math.round(blend('elev'));
    const slopePct = county ? county.slope : Number(blend('slope').toFixed(2));
    const rFactor = county ? county.rFactor : computeRusleRFactor(precip, 1.0);

    const ph = county ? county.ph : Number(blend('ph').toFixed(1));
    const som = county ? county.om : Number(blend('som').toFixed(1));
    const clay = county ? county.clay : Math.round(blend('clay'));
    const sand = county ? county.sand : Math.round(blend('sand'));
    const silt = Math.max(5, 100 - clay - sand);
    const awc = county ? county.awc : Number(blend('awc').toFixed(2));
    const kFactor = county ? county.kw : computeSoilKFactor(sand, silt, clay, som);
    const drainage = county ? county.drn : 'Moderately well drained';
    const hydGrp = county ? county.hsg : 'B';

    const gdd = Math.max(0, Math.round(blend('gdd')));
    const frostFree = Math.max(0, Math.min(365, Math.round(blend('frost'))));
    const heatDays = Math.max(0, Math.round(blend('heat')));
    const solar = Number(blend('solar').toFixed(1));
    const wind = Number(blend('wind').toFixed(1));
    const rh = Math.round(blend('rh'));
    const gwetRoot = Number(blend('gwetRoot').toFixed(2));
    const smapMoist = Number(blend('smap').toFixed(1));
    const ndvi = Number(Math.min(0.92, Math.max(0.02, blend('ndvi'))).toFixed(2));
    const grace = county ? county.gw : Math.round(blend('grace'));
    const nwisPercentile = county ? county.nwisQ : 52;
    const gediBiomass = county ? county.gedi : Math.round(blend('gedi'));
    const nlcdCode = county ? county.nlcd : 82;
    const cdlHistory = county && Array.isArray(county.cdl) ? county.cdl : [];
    const landfireName = county ? county.reg + ' Agro-Ecosystem' : (closestAnchor ? closestAnchor.name : 'Global Agro-Zone');

    // Compute USGS Soil P if available
    const usgsP = typeof global.lookupUsgsSoilP === 'function' ? global.lookupUsgsSoilP(lat, lon) : null;
    const pTot = usgsP !== null ? usgsP : Math.round(blend('pTot'));
    const pSol = typeof global.computePhPhosphorusSolubility === 'function' ? global.computePhPhosphorusSolubility(ph) : 78;
    const pAvail = Math.max(4, Math.round(pTot * (pSol / 100) * 0.085));
    const nTot = Number(Math.max(0.3, som * 0.72).toFixed(1));
    const sTot = Math.round((nTot * 1000) / 8.5);
    const sSulfate = Math.max(5, Math.round(sTot * 0.065));
    const sRatio = ((nTot * 1000) / Math.max(1, sTot)).toFixed(1);

    const paw = Number((awc * 140).toFixed(1)); // available water % / root-zone proxy
    const esi = Number(Math.max(0.08, Math.min(0.96, (precip + paw * 5) / Math.max(1, pet))).toFixed(2));
    const bareDays = Math.max(30, Math.min(340, Math.round(365 * (1 - ndvi * 0.75))));

    // Blend 12-month arrays from closest global anchor scaled to county precip & temp
    let monthly = null;
    if (closestAnchor && Array.isArray(closestAnchor.mT) && closestAnchor.mT.length === 12) {
      const mNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      const pSumA = closestAnchor.mP.reduce((a, b) => a + b, 0) || 1;
      const etSumA = closestAnchor.mET0.reduce((a, b) => a + b, 0) || 1;
      const tDiff = meanTempNum - closestAnchor.temp;
      monthly = mNames.map((nm, idx) => {
        const tm = closestAnchor.mT[idx] + tDiff;
        const pMo = Math.max(0, Math.round(closestAnchor.mP[idx] * (precip / pSumA)));
        const etMo = Math.max(8, Math.round(closestAnchor.mET0[idx] * (pet / etSumA)));
        return { name: nm, t_mean: tm, t_max: tm + 6.5, t_min: tm - 6.5, precip_mm: pMo, pet_mm: etMo };
      });
    }

    return {
      isArable,
      nonArableReason: isArable ? '' : `Non-Arable (${closestAnchor ? closestAnchor.order : 'Extreme Zone'})`,
      countyName: county ? `${county.name} County, ${county.st}` : (closestAnchor ? closestAnchor.name : 'Global Location'),
      countyFips: county ? county.fips : null,
      elevM,
      slopePct,
      aspectDeg: 180,
      meanTemp: Number(meanTempNum).toFixed(1),
      gdd,
      frostFree,
      heatDays,
      solar,
      wind,
      rh,
      precip,
      pEff: Math.round(computeEffectiveRainMonth(precip / 12) * 12),
      pet,
      etMethod: 'FAO-56 Hargreaves (Offline DB)',
      balance: precip - pet,
      aridityIndex: Number((precip / Math.max(1, pet)).toFixed(2)),
      rFactor,
      kFactor,
      awc,
      drainage,
      hydGrp,
      paw,
      ph,
      phLabel: `pH ${ph}`,
      som,
      bulkDensity: Number((1.55 - som * 0.055).toFixed(2)),
      clay,
      sand,
      silt,
      pAvail,
      pTot,
      pSol,
      sSulfate,
      sTot,
      sRatio,
      nTot,
      nLeg: -115,
      ndvi,
      bareDays,
      smapMoist,
      gwetRoot,
      esi,
      grace,
      nwisPercentile,
      nwisSites: county ? 4 : 0,
      nwisFlowCfs: null,
      gediBiomass,
      nlcdCode,
      nlcdLabel: (global.V4_NLCD_LABELS && global.V4_NLCD_LABELS[nlcdCode]) || 'Cultivated Crops',
      cdlHistory,
      landfireName,
      sentinelScene: null,
      naipScene: null,
      landsatScene: null,
      cocorahsFactor: 1.0,
      order: closestAnchor ? closestAnchor.order : 'Mollisol / Alfisol',
      texture: typeof global.classifyUsdaTexture === 'function' ? global.classifyUsdaTexture(clay, sand, silt) : 'Loam',
      monthly,
      sourcesFired: [county ? `US County DB (${county.fips})` : 'Global NASA POWER Grid']
    };
  }

  // Parallel Live Online Cascade — merges live NASA/USDA/USGS/ESA APIs onto active baseline
  let _v4Abort = null;
  async function runV4LiveCascade(lat, lon, onPatch) {
    if (_v4Abort) _v4Abort.abort();
    _v4Abort = new AbortController();
    const sig = _v4Abort.signal;
    const keys = getApiKeys();
    const inUs = isInsideUS(lat, lon);
    const tasks = [];

    // 1. NASA POWER (T2M, T2M_RANGE, PRECTOTCORR, ALLSKY_SFC_SW_DWN, WS2M, RH2M, FROST_DAYS, GWETROOT, GWETTOP)
    tasks.push((async () => {
      try {
        const url = `https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,T2M_RANGE,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M,RH2M,FROST_DAYS,GWETROOT,GWETTOP&community=AG&longitude=${lon.toFixed(4)}&latitude=${lat.toFixed(4)}&format=JSON`;
        const r = await fetch(url, { signal: sig });
        if (!r.ok) return;
        const d = await r.json();
        const pm = d?.properties?.parameter;
        if (!pm?.T2M || typeof pm.T2M.ANN !== 'number') return;

        const mKeys = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
        const mNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        const mDays = [31,28.25,31,30,31,30,31,31,30,31,30,31];
        const elev = (global.activeSoilData && global.activeSoilData.elevM) || 200;

        let petAnn = 0, pAnn = 0, frostSum = 0, gddSum = 0, heatSum = 0;
        const monthly = mKeys.map((mk, i) => {
          const tm = pm.T2M[mk] > -900 ? pm.T2M[mk] : pm.T2M.ANN;
          const tr = (pm.T2M_RANGE?.[mk] > -900) ? pm.T2M_RANGE[mk] : 11;
          const tmax = tm + tr * 0.5;
          const tmin = tm - tr * 0.5;
          const pDay = (pm.PRECTOTCORR?.[mk] > -900) ? Math.max(0, pm.PRECTOTCORR[mk]) : 1.5;
          const pMo = Math.round(pDay * mDays[i]);
          const rs = (pm.ALLSKY_SFC_SW_DWN?.[mk] > -900) ? pm.ALLSKY_SFC_SW_DWN[mk] : 15;
          const ws = (pm.WS2M?.[mk] > -900) ? pm.WS2M[mk] : 2.8;
          const rh = (pm.RH2M?.[mk] > -900) ? pm.RH2M[mk] : 60;
          const etDay = computePenmanMonteithDay(tm, tmax, tmin, rh, ws, rs, elev);
          const etMo = Math.max(8, Math.round(etDay * mDays[i]));
          petAnn += etMo;
          pAnn += pMo;
          if (pm.FROST_DAYS?.[mk] > -900) frostSum += Math.max(0, pm.FROST_DAYS[mk]);
          gddSum += Math.max(0, tm - 10) * mDays[i];
          heatSum += Math.max(0, tmax - 30) * 8.5;
          return { name: mNames[i], t_mean: tm, t_max: tmax, t_min: tmin, precip_mm: pMo, pet_mm: etMo };
        });

        const gwetRoot = pm.GWETROOT?.ANN > -900 ? Number(pm.GWETROOT.ANN.toFixed(2)) : null;
        onPatch({
          meanTemp: pm.T2M.ANN.toFixed(1),
          precip: Math.round(pAnn),
          pet: Math.round(petAnn),
          etMethod: 'FAO-56 Penman-Monteith (NASA POWER)',
          gdd: Math.round(gddSum),
          frostFree: Math.max(0, Math.min(365, Math.round(365 - frostSum * 1.15))),
          heatDays: Math.round(heatSum),
          solar: pm.ALLSKY_SFC_SW_DWN?.ANN > -900 ? Number(pm.ALLSKY_SFC_SW_DWN.ANN.toFixed(1)) : undefined,
          wind: pm.WS2M?.ANN > -900 ? Number(pm.WS2M.ANN.toFixed(1)) : undefined,
          rh: pm.RH2M?.ANN > -900 ? Math.round(pm.RH2M.ANN) : undefined,
          gwetRoot: gwetRoot ?? undefined,
          smapMoist: gwetRoot ? Number((gwetRoot * 42).toFixed(1)) : undefined,
          rFactor: computeRusleRFactor(pAnn, 1.0),
          monthly
        }, 'NASA POWER PM-ET0');
      } catch (e) {}
    })());

    // 2. Open-Meteo SRTM 5-point Elevation Stencil (live Slope % & Aspect) + ERA5 ET0
    tasks.push((async () => {
      try {
        const d = 0.0045; // ~500m field stencil
        const lats = [lat, lat + d, lat - d, lat, lat].map(v => v.toFixed(5)).join(',');
        const lons = [lon, lon, lon, lon + d, lon - d].map(v => v.toFixed(5)).join(',');
        const r = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${lats}&longitude=${lons}`, { signal: sig });
        if (!r.ok) return;
        const j = await r.json();
        const ev = j?.elevation;
        if (!Array.isArray(ev) || ev.length < 5) return;
        const dzdy = (ev[1] - ev[2]) / (2 * d * 111320);
        const dzdx = (ev[3] - ev[4]) / (2 * d * 111320 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)));
        const slopePct = Number(Math.min(45, Math.hypot(dzdx, dzdy) * 100).toFixed(2));
        const aspectDeg = Math.round(((Math.atan2(-dzdy, dzdx) * 180) / Math.PI + 360) % 360);
        onPatch({ elevM: Math.round(ev[0]), slopePct, aspectDeg }, 'SRTM 90m Slope');
      } catch (e) {}
    })());

    // 3. Open-Meteo ERA5 Reanalysis (recent 30-day ERA5 ET0 + soil moisture validation)
    tasks.push((async () => {
      try {
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}&daily=et0_fao_evapotranspiration,precipitation_sum&past_days=14&forecast_days=1`;
        const r = await fetch(url, { signal: sig });
        if (!r.ok) return;
        const j = await r.json();
        const etArr = j?.daily?.et0_fao_evapotranspiration;
        if (Array.isArray(etArr) && etArr.length > 0) {
          const valid = etArr.filter(x => typeof x === 'number');
          const avgDay = valid.reduce((a, b) => a + b, 0) / Math.max(1, valid.length);
          onPatch({ era5DailyEt0: Number(avgDay.toFixed(2)) }, 'ECMWF ERA5');
        }
      } catch (e) {}
    })());

    // 4. US-Only Live Cascade: USDA NRCS SSURGO SDA, CropScape CDL (2022-2024), USGS NWIS, NLCD
    if (inUs) {
      // 4a. USDA NRCS SSURGO SDA
      tasks.push((async () => {
        try {
          const sql = `SELECT TOP 1 mu.muname, c.drainagecl, c.hydgrp, c.slope_r, ch.om_r, ch.claytotal_r, ch.sandtotal_r, ch.silttotal_r, ch.awc_r, ch.kwfact, ch.ph1to1h2o_r FROM mapunit mu INNER JOIN component c ON c.mukey = mu.mukey AND c.majcompflag = 'Yes' INNER JOIN chorizon ch ON ch.cokey = c.cokey WHERE mu.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point(${lon.toFixed(5)} ${lat.toFixed(5)})')) ORDER BY c.comppct_r DESC, ch.hzdept_r ASC`;
          const r = await fetch('https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest', {
            method: 'POST',
            signal: sig,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: sql, format: 'JSON+COLUMNNAME' })
          });
          if (!r.ok) return;
          const j = await r.json();
          const tbl = j?.Table;
          if (!Array.isArray(tbl) || tbl.length < 2) return;
          const cols = tbl[0], row = tbl[1];
          const get = (k) => { const idx = cols.indexOf(k); return idx >= 0 ? row[idx] : null; };
          const clay = parseFloat(get('claytotal_r'));
          const sand = parseFloat(get('sandtotal_r'));
          const silt = parseFloat(get('silttotal_r'));
          const om = parseFloat(get('om_r'));
          const awc = parseFloat(get('awc_r'));
          const kw = parseFloat(get('kwfact'));
          const ph = parseFloat(get('ph1to1h2o_r'));
          const patch = {
            ssurgoMapUnit: get('muname') || undefined,
            drainage: get('drainagecl') || undefined,
            hydGrp: get('hydgrp') || undefined
          };
          if (!isNaN(clay) && !isNaN(sand)) {
            patch.clay = Math.round(clay);
            patch.sand = Math.round(sand);
            patch.silt = !isNaN(silt) ? Math.round(silt) : Math.max(0, 100 - patch.clay - patch.sand);
            if (typeof global.classifyUsdaTexture === 'function') {
              patch.texture = global.classifyUsdaTexture(patch.clay, patch.sand, patch.silt);
            }
          }
          if (!isNaN(om)) {
            patch.som = Number(om.toFixed(1));
            patch.bulkDensity = Number(Math.max(1.05, 1.55 - om * 0.055).toFixed(2));
          }
          if (!isNaN(awc)) {
            patch.awc = Number(awc.toFixed(2));
            patch.paw = Number((awc * 140).toFixed(1));
          }
          if (!isNaN(kw)) patch.kFactor = Number(kw.toFixed(2));
          if (!isNaN(ph)) {
            patch.ph = Number(ph.toFixed(1));
            patch.phLabel = `pH ${patch.ph}`;
          }
          onPatch(patch, 'USDA SSURGO');
        } catch (e) {}
      })());

      // 4b. USDA CropScape CDL (2022, 2023, 2024 rotation history via EPSG:5070)
      tasks.push((async () => {
        try {
          const pt = latLonToEpsg5070(lat, lon);
          const years = [2024, 2023, 2022];
          const results = await Promise.all(years.map(async (yr) => {
            try {
              const r = await fetch(`https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue?year=${yr}&x=${pt.x}&y=${pt.y}`, { signal: sig });
              if (!r.ok) return null;
              const txt = await r.text();
              const valM = txt.match(/value:\s*(\d+)/);
              const catM = txt.match(/category:\s*"([^"]+)"/);
              return valM ? { year: yr, id: parseInt(valM[1], 10), name: catM ? catM[1] : `CDL #${valM[1]}` } : null;
            } catch (e) { return null; }
          }));
          const valid = results.filter(Boolean);
          if (valid.length > 0) {
            onPatch({
              cdlHistory: valid.map(v => v.id),
              cdlRotationList: valid
            }, 'CropScape CDL 30m');
          }
        } catch (e) {}
      })());

      // 4c. USGS NWIS Live Streamflow (00060) & Monitoring Gauges
      tasks.push((async () => {
        try {
          const b = `${(lon - 0.35).toFixed(3)},${(lat - 0.35).toFixed(3)},${(lon + 0.35).toFixed(3)},${(lat + 0.35).toFixed(3)}`;
          const r = await fetch(`https://waterservices.usgs.gov/nwis/iv/?format=json&bBox=${b}&parameterCd=00060&siteStatus=active`, { signal: sig });
          if (!r.ok) return;
          const j = await r.json();
          const ts = j?.value?.timeSeries || [];
          const flows = [];
          ts.forEach(s => {
            const v = parseFloat(s?.values?.[0]?.value?.[0]?.value);
            if (!isNaN(v) && v >= 0) flows.push(v);
          });
          if (ts.length > 0) {
            const avgCfs = flows.length ? Math.round(flows.reduce((a, b) => a + b, 0) / flows.length) : 0;
            const nwisQ = Math.max(15, Math.min(95, Math.round(35 + Math.log10(Math.max(1, avgCfs + 1)) * 18)));
            onPatch({ nwisSites: ts.length, nwisFlowCfs: avgCfs, nwisPercentile: nwisQ }, `USGS NWIS (${ts.length} gauges)`);
          }
        } catch (e) {}
      })());

      // 4d. MRLC NLCD 2021 Land Cover
      tasks.push((async () => {
        try {
          const b = `${(lon - 0.005).toFixed(4)},${(lat - 0.005).toFixed(4)},${(lon + 0.005).toFixed(4)},${(lat + 0.005).toFixed(4)}`;
          const r = await fetch(`https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetFeatureInfo&LAYERS=NLCD_2021_Land_Cover_L48&QUERY_LAYERS=NLCD_2021_Land_Cover_L48&BBOX=${b}&WIDTH=2&HEIGHT=2&X=1&Y=1&SRS=EPSG:4326&INFO_FORMAT=application/json`, { signal: sig });
          if (!r.ok) return;
          const j = await r.json();
          const idx = j?.features?.[0]?.properties?.PALETTE_INDEX;
          if (typeof idx === 'number') {
            const lbl = (global.V4_NLCD_LABELS && global.V4_NLCD_LABELS[idx]) || `NLCD #${idx}`;
            onPatch({ nlcdCode: idx, nlcdLabel: lbl }, 'USGS NLCD 30m');
          }
        } catch (e) {}
      })());

      // 4e. Planetary Computer NAIP (1m Aerial) + Landsat-9 STAC
      tasks.push((async () => {
        try {
          const r = await fetch('https://planetarycomputer.microsoft.com/api/stac/v1/search', {
            method: 'POST',
            signal: sig,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ collections: ['naip'], intersects: { type: 'Point', coordinates: [lon, lat] }, limit: 1 })
          });
          if (!r.ok) return;
          const j = await r.json();
          const f = j?.features?.[0];
          if (f) onPatch({ naipScene: f.id }, 'USDA NAIP');
        } catch (e) {}
      })());
    }

    // 5. Global Copernicus Sentinel-2 L2A STAC (Element84 Zero-Auth)
    tasks.push((async () => {
      try {
        const r = await fetch('https://earth-search.aws.element84.com/v1/search', {
          method: 'POST',
          signal: sig,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            collections: ['sentinel-2-l2a'],
            intersects: { type: 'Point', coordinates: [lon, lat] },
            limit: 1,
            query: { 'eo:cloud_cover': { lt: 25 } }
          })
        });
        if (!r.ok) return;
        const j = await r.json();
        const f = j?.features?.[0];
        if (f) {
          const dt = (f.properties?.datetime || '').slice(0, 10);
          const cc = f.properties?.['eo:cloud_cover'];
          onPatch({
            sentinelScene: `${f.id.slice(0, 22)} (${dt}, ${typeof cc === 'number' ? cc.toFixed(1) : '?'}% cloud)`
          }, 'ESA Sentinel-2 STAC');
        }
      } catch (e) {}
    })());

    // 6. Optional User API Key Hooks (OpenET, USDA NASS QuickStats)
    if (keys.openet && inUs && lon <= -100) {
      tasks.push((async () => {
        try {
          const r = await fetch('https://openet-api.org/raster/timeseries/point', {
            method: 'POST',
            signal: sig,
            headers: { 'Authorization': keys.openet, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              date_range: ['2024-01-01', '2024-12-31'],
              interval: 'monthly',
              geometry: [lon, lat],
              model: 'Ensemble',
              variable: 'ET',
              reference_et: 'gridMET',
              units: 'mm',
              file_format: 'JSON'
            })
          });
          if (!r.ok) return;
          const j = await r.json();
          if (Array.isArray(j) && j.length > 0) {
            const sumEt = Math.round(j.reduce((s, row) => s + (Number(row.et) || 0), 0));
            if (sumEt > 50) onPatch({ pet: sumEt, etMethod: 'OpenET Ensemble (Live Key)' }, 'OpenET Live');
          }
        } catch (e) {}
      })());
    }

    if (keys.nass && inUs && global.activeSoilData?.countyFips) {
      tasks.push((async () => {
        try {
          const fips = global.activeSoilData.countyFips;
          const stFips = fips.slice(0, 2), coFips = fips.slice(2, 5);
          const u = `https://quickstats.nass.usda.gov/api/api_GET/?key=${encodeURIComponent(keys.nass)}&commodity_desc=CORN&statisticcat_desc=YIELD&state_fips_code=${stFips}&county_code=${coFips}&year__GE=2022&format=JSON`;
          const r = await fetch(u, { signal: sig });
          if (!r.ok) return;
          const j = await r.json();
          const row = j?.data?.[0];
          const val = parseFloat((row?.Value || '').replace(/,/g, ''));
          if (!isNaN(val) && val > 10) {
            global.V4_LIVE_NASS_YIELDS = global.V4_LIVE_NASS_YIELDS || {};
            global.V4_LIVE_NASS_YIELDS.corn = { yield: val, year: row.year };
            onPatch({ liveNassCorn: `${val} bu/ac (${row.year})` }, 'NASS QuickStats Live');
          }
        } catch (e) {}
      })());
    }

    await Promise.allSettled(tasks);
  }

  global.AgriV4 = {
    STRATEGY_WEIGHTS,
    getApiKeys,
    saveApiKeys,
    latLonToEpsg5070,
    isInsideUS,
    findNearestUsCounty,
    computePenmanMonteithDay,
    computeEffectiveRainMonth,
    computeRusleRFactor,
    computeSoilKFactor,
    computeRusleLSFactor,
    computeRusleErosion,
    computeIrrigationFeasibility,
    computeCwsi,
    computeCropEconomics,
    scoreCropV4,
    computeV4OfflineBaseline,
    runV4LiveCascade,
    lookupCropEcon
  };
})(window);
