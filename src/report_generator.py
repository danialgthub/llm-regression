# 🔹 How It Works
# Converts each PNG chart into a base64 string.

# Embeds them directly in the HTML as <img src="data:image/png;base64,...">.

# No external file dependencies — the report is self‑contained.

# Slack alerts can link to this HTML file, and teammates will see charts inline.


import json
from pathlib import Path
import base64

def embed_image_base64(path):
    """Convert image file to base64 string for embedding in HTML."""
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_html_report(run_id, score, results, diffs=None):
    # Embed charts
    acc_trend_img = embed_image_base64("results/accuracy_trend.png")
    cm_img = embed_image_base64("results/confusion_matrix.png")
    dashboard_img = embed_image_base64("results/dashboard.png")

    html = f"""
    <html>
    <head><title>Eval Report {run_id}</title></head>
    <body>
        <h1>Eval Report {run_id}</h1>
        <h2>Summary</h2>
        <p>Accuracy: {score['accuracy']*100:.2f}% ({score['correct']}/{score['total']})</p>
        <p>Avg relevance: {score['avg_relevance']:.2f}</p>
        <p>Avg latency: {score['avg_latency']:.2f}s</p>
        <p>Avg tokens: {score['avg_tokens']:.2f}</p>
    """

    # Case results table
    html += """
        <h2>Case Results</h2>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Expected</th><th>Actual</th><th>Latency</th><th>Relevance</th></tr>
    """
    for r in results:
        html += f"<tr><td>{r['id']}</td><td>{r['expected']}</td><td>{r['actual']}</td><td>{r['latency']:.2f}s</td><td>{r.get('relevance_score','-')}</td></tr>"
    html += "</table>"

    # Diffs section
    if diffs:
        html += "<h2>Diffs vs Previous Run</h2><ul>"
        for d in diffs:
            if d["category_changed"] or d["summary_changed"]:
                html += f"<li>Case {d['id']} changed: {d['old']} → {d['new']}</li>"
        html += "</ul>"

    # Embedded charts
    if acc_trend_img:
        html += f"<h2>Accuracy Trend</h2><img src='data:image/png;base64,{acc_trend_img}' width='600'>"
    if cm_img:
        html += f"<h2>Confusion Matrix</h2><img src='data:image/png;base64,{cm_img}' width='600'>"
    if dashboard_img:
        html += f"<h2>Dashboard</h2><img src='data:image/png;base64,{dashboard_img}' width='800'>"

    html += "</body></html>"

    Path("results").mkdir(exist_ok=True)
    report_path = f"results/report_{run_id}.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {report_path}")
