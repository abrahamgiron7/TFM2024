import sqlite3

#Fichero que inicializa la base de datos e inserta.

connection = sqlite3.connect('DataBase/Alumn_Data.db')


with open('DataBase/schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO Alumn_Data (email, user, pass1, pass2) VALUES (?, ?, ?, ?)",
            ('abraham1@gmail.com', 'agr', 'contraseña1', 'contraseña2')
            )

cur.execute("INSERT INTO Alumn_Data (email, user, pass1, pass2) VALUES (?, ?, ?, ?)",
            ('abraham2@gmail.com', 'agr2', 'contraseña3', 'contraseña4')
            )

connection.commit()
connection.close()