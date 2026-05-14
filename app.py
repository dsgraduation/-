from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # السماح للواجهة بالتواصل مع السيرفر المحلي
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "FINAL_DATASET.csv")

df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

print("DATASET PATH:", DATASET_PATH)
print("ROWS:", len(df))
print("COLUMNS:", df.columns.tolist())
print("REGIONS SAMPLE:", df["region"].dropna().unique()[:10])

# =========================
# تحميل وتنظيف الداتا
# =========================
df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")

text_columns = [
    "region",
    "city",
    "business_activity",
    "business_examples",
    "competition_level",
    "required_permits",
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

required_numeric = ["avg_capital", "avg_rent_monthly_est", "operating_costs_monthly"]

for col in required_numeric:
    if col not in df.columns:
        raise ValueError(f"العمود غير موجود في الداتا: {col}")
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

if "rent_source" not in df.columns:
    df["rent_source"] = "unknown"

if "business_examples" not in df.columns:
    df["business_examples"] = ""

if "required_permits" not in df.columns:
    df["required_permits"] = ""

# حذف الصفوف الأساسية غير الصالحة فقط
df = df.dropna(subset=["region", "city", "business_activity"])


# =========================
# دوال مساعدة
# =========================
def to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def split_pipe(value):
    # تقسيم النصوص المكتوبة بهذا الشكل: مثال 1 | مثال 2 | مثال 3
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value.lower() in ["nan", "none", ""]:
        return []
    return [x.strip() for x in value.split("|") if x.strip()]


def inverse_score(series):
    # الأقل أفضل، لذلك نعكس الدرجة: أقل إيجار/تشغيل يأخذ درجة أعلى
    max_value = series.max()

    if pd.isna(max_value) or max_value <= 0:
        return pd.Series([50.0] * len(series), index=series.index)

    return ((1 - (series / max_value)) * 100).clip(lower=0, upper=100)


# =========================
# الصفحة الرئيسية
# =========================
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


# =========================
# API: المناطق
# =========================
@app.route("/api/regions", methods=["GET"])
def get_regions():
    regions = sorted(df["region"].dropna().unique().tolist())
    return jsonify({"regions": regions})


# =========================
# API: المدن حسب المنطقة
# =========================
@app.route("/api/cities", methods=["GET"])
def get_cities():
    region = request.args.get("region")

    if region:
        cities = sorted(df[df["region"] == region]["city"].dropna().unique().tolist())
    else:
        cities = sorted(df["city"].dropna().unique().tolist())

    return jsonify({"cities": cities})


# =========================
# API: التوصيات
# =========================
@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def recommend():
    if request.method == "OPTIONS":
        return jsonify({})

    data = request.get_json(force=True) or {}

    region = str(data.get("region", "")).strip()
    city = str(data.get("city", "")).strip()
    capital = to_float(data.get("capital", 0), 0)
    offset = int(data.get("offset", 0) or 0)
    competition_filter = str(data.get("competition_filter", "all")).strip()
    limit = 10

    city_data = df[(df["region"] == region) & (df["city"] == city)].copy()

    if city_data.empty:
        return jsonify({"error": "لا توجد بيانات لهذه المنطقة والمدينة."}), 400

    # =========================
    # عوامل التقييم
    # =========================
    comp_map = {
        "Low": 100,
        "Medium": 60,
        "High": 30,
    }

    city_data["competition_value"] = city_data["competition_level"].map(comp_map).fillna(60)
    city_data["rent_value"] = inverse_score(city_data["avg_rent_monthly_est"])
    city_data["operating_value"] = inverse_score(city_data["operating_costs_monthly"])

    # ملاءمة رأس المال فقط: إذا رأس المال يغطي النشاط تكون 100%
    city_data["capital_fit"] = np.where(
        city_data["avg_capital"] > 0,
        (capital / city_data["avg_capital"] * 100).clip(upper=100),
        0,
    )

    max_capital = city_data["avg_capital"].max()
    min_capital = city_data["avg_capital"].min()

    # =========================
    # وضع رأس المال العالي
    # =========================
    if capital >= max_capital:
        mode = "premium"

        message = (
            "وضع رأس المال العالي\n"
            "رأس المال المدخل يغطي جميع الأنشطة في هذه المدينة، لذلك نسبة الملاءمة المالية لجميع الأنشطة هي 100%، "
            "ثم تم ترتيب النتائج حسب انخفاض المنافسة والإيجار وتكاليف التشغيل."
        )

        rec_data = city_data.copy()

        # score يستخدم للترتيب فقط، وليس لعرض نسبة ملاءمة رأس المال
        rec_data["score"] = (
                0.40 * rec_data["competition_value"]
                + 0.30 * rec_data["rent_value"]
                + 0.30 * rec_data["operating_value"]
        )

        rec_data["suitability_pct"] = 100

    # =========================
    # الوضع الطبيعي
    # =========================
    elif capital >= min_capital:
        mode = "normal"

        message = (
            "نتائج البحث\n"
            "تم عرض الأنشطة التي يقع متوسط رأس المال المطلوب لها ضمن رأس المال المدخل، "
            "ثم ترتيبها حسب الملاءمة العامة بناءً على رأس المال والمنافسة والإيجار وتكاليف التشغيل."
        )

        rec_data = city_data[city_data["avg_capital"] <= capital].copy()

        # score يستخدم لترتيب الأنشطة المناسبة فقط
        rec_data["score"] = (
                0.50 * rec_data["capital_fit"]
                + 0.20 * rec_data["competition_value"]
                + 0.15 * rec_data["rent_value"]
                + 0.15 * rec_data["operating_value"]
        )

        # نسبة الملاءمة المعروضة = ملاءمة رأس المال فقط
        rec_data["suitability_pct"] = rec_data["capital_fit"]

    # =========================
    # وضع رأس المال المنخفض
    # =========================
    else:
        mode = "low"

        message = (
            "رأس المال المدخل أقل من جميع الأنشطة في هذه المدينة.\n"
            "بدل إيقاف التوصية، تم عرض أقرب الأنشطة الممكنة حسب نسبة تغطية رأس المال "
            "وانخفاض التكاليف ومستوى المنافسة."
        )

        rec_data = city_data.copy()

        # في هذا الوضع سيظهر للواجهة كم ينقص المستخدم لتغطية النشاط
        rec_data["capital_gap"] = (rec_data["avg_capital"] - capital).clip(lower=0)

        rec_data["score"] = (
                0.70 * rec_data["capital_fit"]
                + 0.10 * rec_data["competition_value"]
                + 0.10 * rec_data["rent_value"]
                + 0.10 * rec_data["operating_value"]
        )

        rec_data["suitability_pct"] = rec_data["capital_fit"]

    # =========================
    # حسابات إضافية للعرض
    # =========================
    if competition_filter in ["Low", "Medium", "High"]:
        rec_data = rec_data[rec_data["competition_level"] == competition_filter].copy()
    rec_data["total_monthly_cost"] = rec_data["avg_rent_monthly_est"] + rec_data["operating_costs_monthly"]
    rec_data["surplus_capital"] = (capital - rec_data["avg_capital"]).clip(lower=0)

    if "capital_gap" not in rec_data.columns:
        rec_data["capital_gap"] = (rec_data["avg_capital"] - capital).clip(lower=0)

    # ترتيب النتائج حسب score ثم الأقل إيجارًا
    rec_data = rec_data.sort_values(["score", "avg_rent_monthly_est"], ascending=[False, True])

    total = len(rec_data)
    offset = max(0, min(offset, max(total - 1, 0))) if total else 0
    offset = (offset // limit) * limit

    page = rec_data.iloc[offset: offset + limit]
    next_offset = offset + len(page)
    prev_offset = max(0, offset - limit)
    has_more = next_offset < total
    has_prev = offset > 0

    # =========================
    # تجهيز النتائج للواجهة
    # =========================
    results = []

    for _, row in page.iterrows():
        results.append({
            "business_activity": str(row["business_activity"]),
            "business_activity_examples": split_pipe(row.get("business_examples", "")),
            "competition_level": str(row.get("competition_level", "Medium")).strip(),
            "avg_capital": to_float(row.get("avg_capital", 0)),
            "avg_rent_monthly": to_float(row.get("avg_rent_monthly_est", 0)),
            "operating_costs_monthly": to_float(row.get("operating_costs_monthly", 0)),
            "total_monthly_cost": to_float(row.get("total_monthly_cost", 0)),
            "suitability_pct": to_float(row.get("suitability_pct", 0)),
            "surplus_capital": to_float(row.get("surplus_capital", 0)),
            "capital_gap": to_float(row.get("capital_gap", 0)),
            "required_permits": split_pipe(row.get("required_permits", "")),
        })

    return jsonify({
        "region": region,
        "city": city,
        "capital": capital,
        "mode": mode,
        "message": message,
        "total": total,
        "results": results,
        "offset": offset,
        "prev_offset": prev_offset,
        "next_offset": next_offset,
        "has_prev": has_prev,
        "has_more": has_more,
        "limit": limit,
    })


if __name__ == "__main__":
    app.run(debug=True)

