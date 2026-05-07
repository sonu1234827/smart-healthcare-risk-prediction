import sqlite3
from datetime import datetime
import struct

DB_PATH = "database.db"

# -------------------------
# SAFE INT
# -------------------------
def _safe_int(v):
    """Convert anything to a plain Python int safely."""

    try:
        # 🔥 handle strings like "1.0"
        if isinstance(v, str):
            return int(float(v))

        # normal numbers
        if isinstance(v, (int, float)):
            return int(v)

        # bytes handling
        if isinstance(v, bytes):
            if len(v) == 8:
                return struct.unpack('<q', v)[0]
            if len(v) == 4:
                return struct.unpack('<i', v)[0]

            try:
                return int(float(v.decode().strip()))
            except:
                return 0

        # fallback
        return int(float(v))

    except:
        return 0


# -------------------------
# SAFE FLOAT
# -------------------------
def _safe_float(v):
    """Convert anything to a plain Python float safely."""

    try:
        if isinstance(v, str):
            return float(v)

        if isinstance(v, (int, float)):
            return float(v)

        if isinstance(v, bytes):
            if len(v) == 8:
                return struct.unpack('<d', v)[0]
            if len(v) == 4:
                return struct.unpack('<f', v)[0]

            try:
                return float(v.decode().strip())
            except:
                return 0.0

        return float(v)

    except:
        return 0.0


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
    """, (
        user,
        name,
        disease,
        _safe_int(pred),     # 🔥 FIXED
        _safe_float(prob),   # 🔥 FIXED
        time
    ))

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

    # 🔥 CLEAN DATA (CRITICAL FIX)
    clean = []
    for row in data:
        try:
            name_, disease_, pred_, prob_, time_ = row

            clean.append((
                name_,
                disease_,
                _safe_int(pred_),     # 🔥 FIX
                _safe_float(prob_),   # 🔥 FIX
                time_
            ))
        except:
            continue  # skip corrupted row

    return clean