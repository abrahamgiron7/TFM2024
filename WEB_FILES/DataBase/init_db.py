 
import sqlite3

#Crea la base de datos si no existe
conn = sqlite3.connect('DataBase/Alumn_Data.db')   

#Ejecuta el schema.sql
with open('DataBase/schema.sql') as f:
    conn.executescript(f.read())

conn.commit()
conn.close()