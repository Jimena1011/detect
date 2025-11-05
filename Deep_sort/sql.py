import psycopg2

try:
    conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1606", port="5432")
    cur = conn.cursor()
    print("✅ Conexión exitosa.")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conteo (
        id SERIAL PRIMARY KEY,
        clase VARCHAR(255),
        speed VARCHAR(255),
        way VARCHAR(255),
        fecha VARCHAR(255),
        camara VARCHAR(255),
        cuenta INT
    );
    """)
    conn.commit()
    print("✅ Tabla creada o ya existente.")

except Exception as e:
    print("❌ Error:", e)
finally:
    cur.close()
    conn.close()
