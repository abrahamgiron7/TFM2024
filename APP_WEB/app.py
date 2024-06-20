from flask import Flask, render_template, redirect, request, url_for, session, make_response
import sqlite3
from werkzeug.exceptions import abort

app = Flask(__name__)

#End points de la app --------------------------

@app.route('/')
def index():
    #Añado para la DB
    conn = get_db_connection()
    datos = conn.execute('SELECT * FROM prueba').fetchall()
    conn.close()
    
    #Si quiero ver directamente las tuplas devueltas descomentar:
    #return [tuple(row) for row in prueba]
    
    #Carga HTML y CSS, y pasa posts con los resultados.
    #return render_template("index.html")
    return render_template("index.html", datos_tabla=datos)
#Sección de Arranque de la app -----------------

def get_db_connection():
    conn = sqlite3.connect('DataBase/database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_reg_by_id(user_id):
    conn = get_db_connection()
    reg = conn.execute('SELECT * FROM prueba WHERE id = ?',
                        (user_id,)).fetchone()
    conn.close()
    if reg is None:
        abort(404)
    return reg

@app.route('/<int:user_id>')
def reg(user_id):
    reg = get_reg_by_id(user_id)
    return render_template('index.html', registro=reg)

""" @app.route("/api/addstreamer", methods=["POST"])
def addstreamer():
    try:
        email = request.form["email"]
        user = request.form["user"]
        pass1 = request.form["pass1"]
        pass2 = request.form["pass2"]

        streamer = Streamers(name, int(subs), int(followers))
        db.session.add(streamer)
        db.session.commit()

        return jsonify(streamer.serialize()), 200

    except Exception:
        exception("\n[SERVER]: Error in route /api/addstreamer. Log: \n")
        return jsonify({"msg": "Algo ha salido mal"}), 500 """

# Route to SELECT all data from the database and display in a table      
@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM students")

    rows = cur.fetchall()
    con.close()
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)

if __name__=='__main__':
    app.run(debug=True,port=3000)

    