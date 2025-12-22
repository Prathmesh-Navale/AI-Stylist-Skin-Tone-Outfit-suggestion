import pymysql

try:
    db = pymysql.connect(
        host="localhost",
        user="toot",       # adjust if you use a different MySQL user
        password="",       # set your MySQL password if you created one
        database="outfit_recommender"
    )
    cur = db.cursor()
    cur.execute("SELECT DATABASE();")
    print("✅ Connected to:", cur.fetchone())
    db.close()
except Exception as e:
    print("❌ Database connection failed:", e)
