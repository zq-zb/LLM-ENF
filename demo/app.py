import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = r"D:\CR\社会计算\2\LLM-ENF"
demo_dir = os.path.join(BASE_DIR, "demo")

sys.path.insert(0, demo_dir)
from engine import RecEngine

app = Flask(
    __name__,
    template_folder=os.path.join(demo_dir, "templates"),
    static_folder=os.path.join(demo_dir, "static")
)

# Initialize engine globally
engine = RecEngine()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    # Map requests to local files. E.g. /static/images/123.jpg maps to D:\CR\社会计算\2\LLM-ENF\final\images\123.jpg
    if filename.startswith("images/"):
        image_name = filename[len("images/"):]
        image_dir = os.path.join(BASE_DIR, "final", "images")
        return send_from_directory(image_dir, image_name)
    return send_from_directory(os.path.join(demo_dir, "static"), filename)

# API ENDPOINTS
@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    domain = request.args.get("domain", "all")
    try:
        limit = int(request.args.get("limit", 30))
    except ValueError:
        limit = 30
    results = engine.search_items(query, domain=domain, limit=limit)
    return jsonify(results)

@app.route("/api/popular/<domain>", methods=["GET"])
def popular(domain):
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    results = engine.get_popular(domain=domain, limit=limit)
    return jsonify(results)

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    items = data.get("items", [])
    target = data.get("target", "book")
    try:
        top_k = int(data.get("top_k", 10))
    except ValueError:
        top_k = 10
    
    # items can be user_id (string/int) or a list of items
    if isinstance(items, (int, str)) or (isinstance(items, list) and len(items) == 1 and str(items[0]).isdigit()):
        user_id = int(items[0]) if isinstance(items, list) else int(items)
        recs = engine.get_recommendations_for_user(user_id, target_domain=target, top_k=top_k)
    else:
        # manual sequence
        item_ids = [int(x) for x in items if str(x).isdigit()]
        recs = engine.get_recommendations_from_sequence(item_ids, target_domain=target, top_k=top_k)
        
    return jsonify(recs)

@app.route("/api/users", methods=["GET"])
def get_users():
    users = []
    for uid in engine.top_50_users:
        profile = engine.get_user_profile(uid)
        users.append({
            "user_id": profile["user_id"],
            "username": profile["username"],
            "history_count": profile["history_count"],
            "movie_count": profile["movie_count"],
            "book_count": profile["book_count"]
        })
    return jsonify(users)

@app.route("/api/user/<int:user_id>", methods=["GET"])
def get_user_detail(user_id):
    try:
        profile = engine.get_user_profile(user_id)
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route("/api/similar_users", methods=["POST"])
def similar_users():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    items = data.get("items", [])
    try:
        top_k = int(data.get("top_k", 5))
    except ValueError:
        top_k = 5

    if user_id is not None:
        user_id = int(user_id)
        target_seq = set([x[1] for x in engine.user_seqs.get(user_id, {}).get("sxy", [])])
    else:
        target_seq = set([int(x) for x in items])

    # Simple overlap/Jaccard similarity with active users
    similarities = []
    for uid, seqs in engine.user_seqs.items():
        if uid == user_id:
            continue
        u_seq = set([x[1] for x in seqs.get("sxy", [])])
        if not u_seq or not target_seq:
            continue
        intersection = target_seq.intersection(u_seq)
        union = target_seq.union(u_seq)
        jaccard = len(intersection) / len(union) if union else 0
        if jaccard > 0:
            similarities.append({
                "user_id": uid,
                "username": engine.id2user.get(uid, f"User {uid}"),
                "similarity": jaccard,
                "overlap_items": [engine.get_item_info(iid) for iid in intersection]
            })

    similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]
    return jsonify(similarities)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
