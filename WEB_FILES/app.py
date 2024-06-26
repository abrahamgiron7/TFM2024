from flask import Flask, render_template, request
import sqlite3
import csv
import os
from predict import run_prediction, train_model   # Importa la función de predicción desde predict.py

app = Flask(__name__)
app.secret_key = 'tfm'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#Función de carga de csv

def cargar_datos_csv(filepath):
    try:
        conn = sqlite3.connect('DataBase/Alumn_Data.db')
        cur = conn.cursor()
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]
        
        cur.executemany(
            "INSERT INTO Alumn_Data ("
            "A_FECHA_NAC, A_PAIS, A_IDIOMA_NVL, P_PAIS, M_PAIS, P_EDAD, M_EDAD, "
            "P_TRABAJO, M_TRABAJO, P_ESTUDIOS, M_ESTUDIOS, VEHICULOS_FAMILIA, A_HERMANOS, "
            "A_RESIDENCIA, FAMILIARES_RESIDENCIA, ESO1_C_1EV, ESO1_C_2EV, ESO1_C_EVF, "
            "ESO1_EF_1EV, ESO1_EF_2EV, ESO1_EF_EVF, ESO1_I_1EV, ESO1_I_2EV, "
            "ESO1_I_EVF, ESO1_M_1EV, ESO1_M_2EV, ESO1_M_EVF, ESO1_MEDIA_1EV, "
            "ESO1_MEDIA_2EV, ESO1_MEDIA_EVF, ESO2_C_1EV, ESO2_C_2EV, ESO2_C_EVF, "
            "ESO2_EF_1EV, ESO2_EF_2EV, ESO2_EF_EVF, ESO2_I_1EV, ESO2_I_2EV, "
            "ESO2_I_EVF, ESO2_M_1EV, ESO2_M_2EV, ESO2_M_EVF, ESO2_MEDIA_1EV, "
            "ESO2_MEDIA_2EV, ESO2_MEDIA_EVF, FPB1_MEDIA_1EV, FPB1_MEDIA_2EV, "
            "FPB1_MEDIA_EVF, FPB1_P_1EV, FPB1_P_2EV, FPB1_P_EVF, FPB1_CA_1EV, "
            "FPB1_CA_2EV, FPB1_CA_EVF, FPB1_CS_1EV, FPB1_CS_2EV, FPB1_CS_EVF, "
            "REP_1ESO, REP_2ESO, REP_1FPB, REC_1EV, REC_2EV, REC_EVEX, PROMOCIONA"
            ") VALUES (:A_FECHA_NAC, :A_PAIS, :A_IDIOMA_NVL, :P_PAIS, :M_PAIS, :P_EDAD, :M_EDAD, "
            ":P_TRABAJO, :M_TRABAJO, :P_ESTUDIOS, :M_ESTUDIOS, :VEHICULOS_FAMILIA, :A_HERMANOS, "
            ":A_RESIDENCIA, :FAMILIARES_RESIDENCIA, :ESO1_C_1EV, :ESO1_C_2EV, :ESO1_C_EVF, "
            ":ESO1_EF_1EV, :ESO1_EF_2EV, :ESO1_EF_EVF, :ESO1_I_1EV, :ESO1_I_2EV, "
            ":ESO1_I_EVF, :ESO1_M_1EV, :ESO1_M_2EV, :ESO1_M_EVF, :ESO1_MEDIA_1EV, "
            ":ESO1_MEDIA_2EV, :ESO1_MEDIA_EVF, :ESO2_C_1EV, :ESO2_C_2EV, :ESO2_C_EVF, "
            ":ESO2_EF_1EV, :ESO2_EF_2EV, :ESO2_EF_EVF, :ESO2_I_1EV, :ESO2_I_2EV, "
            ":ESO2_I_EVF, :ESO2_M_1EV, :ESO2_M_2EV, :ESO2_M_EVF, :ESO2_MEDIA_1EV, "
            ":ESO2_MEDIA_2EV, :ESO2_MEDIA_EVF, :FPB1_MEDIA_1EV, :FPB1_MEDIA_2EV, "
            ":FPB1_MEDIA_EVF, :FPB1_P_1EV, :FPB1_P_2EV, :FPB1_P_EVF, :FPB1_CA_1EV, "
            ":FPB1_CA_2EV, :FPB1_CA_EVF, :FPB1_CS_1EV, :FPB1_CS_2EV, :FPB1_CS_EVF, "
            ":REP_1ESO, :REP_2ESO, :REP_1FPB, :REC_1EV, :REC_2EV, :REC_EVEX, :PROMOCIONA)",
            rows
        )
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        return str(e)        

# Home Page route - Predecir - IA
@app.route("/")
def home():
    """ print('Home') """
    return render_template("predict.html")

#Page Predecir
@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        train_model()  # Ejecuta la función para entrenar el modelo
        return render_template('train_results.html', message="Modelo entrenado correctamente.")
    except Exception as e:
        return render_template('train_error.html', message="Error al entrenar el modelo: " + str(e))

# Page Entrenar - IA
@app.route("/train")
def train():
    """ print('Home') """
    return render_template("train.html")

# Route to Predict with csv     
@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        print("entra predict")

        if 'csvFile' not in request.files:
            return render_template('predict_error.html', message='No se seleccionó ningún archivo.'), 400

        file = request.files['csvFile']
        if file.filename == '':
            return render_template('predict_error.html', message='Archivo vacío.'), 400

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Llamar a la función que realiza la predicción
            result = run_prediction(filepath) 

            return render_template('predict_result.html', predictions=result.to_dict(orient='records'))

        return render_template('predict_error.html', message='Archivo no válido. Por favor, suba un archivo CSV.'), 400

    except Exception as e:
        return render_template('predict_error.html', message=f'Error en la predicción: {str(e)}'), 500

# Route to SELECT all data from the database and display in a table      
@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    """ print('list') """
    con = sqlite3.connect("DataBase/Alumn_Data.db")
    con.row_factory = sqlite3.Row

    cur = con.cursor()
    cur.execute("SELECT id, * FROM Alumn_Data")

    rows = cur.fetchall()
    """ print(rows[1][1]) """
    con.close()
    
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)

# Page carga csv con datos de entrenamiento
@app.route("/enternew", methods=['POST','GET'])
def enternew():
    return render_template("alta.html")

# Route to charge csv with train data in db
@app.route("/addrec", methods = ['POST', 'GET'])
def addrec():
    try:
        if 'csvFile' not in request.files:
            return render_template('alta_error.html', message='No se seleccionó ningún archivo.'), 400

        file = request.files['csvFile']
        if file.filename == '':
            return render_template('alta_error.html', message='Archivo vacío.'), 400

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = cargar_datos_csv(filepath)
            if result == True:
                return render_template('alta_success.html', message='Archivo cargado y datos insertados correctamente.'), 200
            else:
                return render_template('alta_error.html', message=result), 400

        return render_template('alta_error.html', message='Archivo no válido. Por favor, suba un archivo CSV.'), 400

    except Exception as e:
        return render_template('alta_error.html', message=f'Error en la carga y/o inserción de datos: {str(e)}'), 500

# Route used to DELETE a specific record in the database    
@app.route("/delete", methods=['POST','GET'])
def delete():
    if request.method == 'POST':
        try:

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