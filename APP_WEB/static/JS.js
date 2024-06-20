function genera_tabla() {
    // Obtener la referencia del elemento body
    var body = document.getElementById("Consulta");
  
    // Crea un elemento <table> y un elemento <tbody>
    var tabla = document.createElement("table");
    //Mete estilos en la tabla que va generando
    tabla.style.cssText = 'background-color: white;';

    var tblBody = document.createElement("tbody");
  
    //Creo el encabezado - Luego lo leere de la DB con un for
    var fila_header=document.createElement("tr");
    fila_header.setAttribute("id", "Cabecera");
    var columna_header1=document.createElement("td");
    var columna_header2=document.createElement("td");
    var columna_header3=document.createElement("td");
    var columna_header4=document.createElement("td");
    columna_header1.appendChild(document.createTextNode("Email"));
    columna_header2.appendChild(document.createTextNode("Username"));
    columna_header3.appendChild(document.createTextNode("Password_1"));
    columna_header4.appendChild(document.createTextNode("Password_2"));
    fila_header.appendChild(columna_header1);
    fila_header.appendChild(columna_header2);
    fila_header.appendChild(columna_header3);
    fila_header.appendChild(columna_header4);
    
    tblBody.appendChild(fila_header);


    //Añado id para cambio estilo del encabezado en el CSS
    document.getElementById("Cabecera")
/*     cabecera.style.color = 'red'; */

    // Crea las celdas
    for (var i = 0; i < 2; i++) {
      // Crea las hileras de la tabla
      var hilera = document.createElement("tr");
  
      for (var j = 0; j < 4; j++) {
        // Crea un elemento <td> y un nodo de texto, haz que el nodo de
        // texto sea el contenido de <td>, ubica el elemento <td> al final
        // de la hilera de la tabla
        var celda = document.createElement("td");
        var textoCelda = document.createTextNode(
          "celda en la hilera " + i + ", columna " + j,
        );
        celda.appendChild(textoCelda);
        hilera.appendChild(celda);
      }
  
      // agrega la hilera al final de la tabla (al final del elemento tblbody)
      tblBody.appendChild(hilera);
    }
  
    // posiciona el <tbody> debajo del elemento <table>
    tabla.appendChild(tblBody);
    // appends <table> into <body>
    body.appendChild(tabla);
    // modifica el atributo "border" de la tabla y lo fija a "2";
    tabla.setAttribute("border", "2");
}

function cargar_tabla(){
    // Obtener la referencia del elemento body
    var body = document.getElementsByTagName("Consulta")[0];
  
    // Crea un elemento <table> y un elemento <tbody>
    var tabla = document.createElement("table");
    //Mete estilos en la tabla que va generando
    tabla.style.cssText = 'background-color: white;';

    var tblBody = document.createElement("tbody");
  
    // Crea las celdas
    for (var i = 0; i < 2; i++) {
      // Crea las hileras de la tabla
      var hilera = document.createElement("tr");
  
      for (var j = 0; j < 4; j++) {
        // Crea un elemento <td> y un nodo de texto, haz que el nodo de
        // texto sea el contenido de <td>, ubica el elemento <td> al final
        // de la hilera de la tabla
        var celda = document.createElement("td");
        var textoCelda = document.createTextNode(
          "celda en la hilera " + i + ", columna " + j,
        );
        celda.appendChild(textoCelda);
        hilera.appendChild(celda);
      }
  
      // agrega la hilera al final de la tabla (al final del elemento tblbody)
      tblBody.appendChild(hilera);
    }
  
    // posiciona el <tbody> debajo del elemento <table>
    tabla.appendChild(tblBody);
    // appends <table> into <body>
    body.appendChild(tabla);
    // modifica el atributo "border" de la tabla y lo fija a "2";
    tabla.setAttribute("border", "2");    

}



function anadir_registro(){
    

}
