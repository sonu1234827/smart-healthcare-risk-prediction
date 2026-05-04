import sqlite3
from datetime import datetime

DB_PATH = "database.db"

# -------------------------
# INIT DB
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        user TEXT,
        name TEXT,
        disease TEXT,
        prediction INTEGER,
        probability REAL,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()

# -------------------------
# REGISTER
# -------------------------
def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

# -------------------------
# LOGIN
# -------------------------
def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    data = c.fetchone()
    conn.close()
    return data is not None

# -------------------------
# SAVE HISTORY
# -------------------------
def save_history(user, name, disease, pred, prob):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
    INSERT INTO history (user, name, disease, prediction, probability, time)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (user, name, disease, pred, prob, time))

    conn.commit()
    conn.close()

# -------------------------
# GET HISTORY
# -------------------------
def get_history(user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    SELECT name, disease, prediction, probability, time
    FROM history
    WHERE user=?
    ORDER BY time
    """, (user,))

    data = c.fetchall()
    conn.close()
    return data