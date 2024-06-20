from flask import Flask, render_template, redirect, request, url_for, session, make_response
import sqlite3
from werkzeug.exceptions import abort

app = Flask(__name__)

#End points de la app --------------------------

""" @app.route('/')
def index():
    #Añado para la DB
    conn = get_db_connection()
    datos = conn.execute('SELECT * FROM Alumn_Data').fetchall()
    conn.close()
    
    #Si quiero ver directamente las tuplas devueltas descomentar:
    #return [tuple(row) for row in prueba]
    
    #Carga HTML y CSS, y pasa posts con los resultados.
    #return render_template("index.html")
    return render_template("index.html", datos_tabla=datos)
#Sección de Arranque de la app ----------------- """

""" def get_db_connection():
    conn = sqlite3.connect('DataBase/Alumn_Data.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_reg_by_id(user_id):
    conn = get_db_connection()
    reg = conn.execute('SELECT * FROM Alumn_Data WHERE id = ?',
                        (user_id,)).fetchone()
    conn.close()
    if reg is None:
        abort(404)
    return reg """

""" @app.route('/<int:user_id>')
def reg(user_id):
    reg = get_reg_by_id(user_id)
    return render_template('index.html', registro=reg) """

# Home Page route - Predecir - IA
@app.route("/")
def home():
    """ print('Home') """
    return render_template("predict.html")


# Route to SELECT all data from the database and display in a table      
@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    """ print('list') """
    con = sqlite3.connect("DataBase/Alumn_Data.db")
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM Alumn_Data")

    rows = cur.fetchall()
    """ print(rows[1][1]) """
    con.close()
    
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)

# Route to form used to add a new student to the database
@app.route("/enternew", methods=['POST','GET'])
def enternew():
    return render_template("alta.html")

# Route to add a new record (INSERT) student data to the database - Mensaje de que se ha insertado correctamente
@app.route("/addrec", methods = ['POST', 'GET'])
def addrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            print('try')
            email = request.form['email']
            user = request.form['user']
            pass1 = request.form['pass1']
            pass2 = request.form['pass2']

            # Connect to SQLite3 database and execute the INSERT
            with sqlite3.connect('DataBase/Alumn_Data.db') as con:
                cur = con.cursor()
                cur.execute("INSERT INTO Alumn_Data (email, user, pass1, pass2) VALUES (?,?,?,?)",(email, user, pass1, pass2))

                con.commit()
                msg = "Record successfully added to database"
        except:
            print('except')
            con.rollback()
            msg = "Error in the INSERT"

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

# Route that will SELECT a specific row in the database then load an Edit form 
@app.route("/edit", methods=['POST','GET'])
def edit():
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            id = request.form['id']
            # Connect to the database and SELECT a specific rowid
            con = sqlite3.connect("DataBase/Alumn_Data.db")
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            cur.execute("SELECT rowid, * FROM Alumn_Data WHERE rowid = " + id)

            rows = cur.fetchall()
        except:
            id=None
        finally:
            con.close()
            # Send the specific record of data to edit.html
            return render_template("edit.html",rows=rows)

# Route used to execute the UPDATE statement on a specific record in the database - Mensaje de que se ha editado
@app.route("/editrec", methods=['POST','GET'])
def editrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['id']
            email = request.form['email']
            user = request.form['user']
            pass1 = request.form['pass1']
            pass2 = request.form['pass2']
            
            # UPDATE a specific record in the database based on the rowid
            with sqlite3.connect('DataBase/Alumn_Data.db') as con:
                cur = con.cursor()
                cur.execute("UPDATE Alumn_Data SET email='"+email+"', user='"+user+"', pass1='"+pass1+"', pass2='"+pass2+"' WHERE id="+rowid)

                con.commit()
                msg = "Record successfully edited in the database"
        except:
            con.rollback()
            msg = "Error in the Edit: UPDATE Alumn_Data SET name="+email+", addr="+user+", city="+pass1+", zip="+pass2+" WHERE rowid="+rowid

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

# Route used to DELETE a specific record in the database    
@app.route("/delete", methods=['POST','GET'])
def delete():
    if request.method == 'POST':
        try:
             # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['id']
            # Connect to the database and DELETE a specific record based on rowid
            with sqlite3.connect('DataBase/Alumn_Data.db') as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM Alumn_Data WHERE rowid="+rowid)

                    con.commit()
                    msg = "Record successfully deleted from the database"
        except:
            con.rollback()
            msg = "Error in the DELETE"

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

if __name__=='__main__':
    app.run(debug=True,port=3000)

