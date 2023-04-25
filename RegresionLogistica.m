clear all
close all

% --------------------- Obtencion de datos de entrenamiento
datos = readmatrix("dataset_RegresionLogistica1.csv");
[m, n] = size(datos); 
x = datos(:, 1:n-1); 
y = datos(:, n);

% --------------------- Impresion de los datos de entrenamiento
figure('Name','Datos de entrenamiento crudos');
figure(1);
hold on;
clase0 = find(y==0); 
clase1 = find(y==1); 
plot(x(clase0,1), x(clase0, 2), "ok", "MarkerFaceColor", "c"); 
plot(x(clase1,1), x(clase1, 2), "dk", "MarkerFaceColor", "m");
title("Calificaciones de alumnos");
xlabel("x1 = Parcial 1");
ylabel("x2 = Parcial 2");

% --------------------- Estandarizacion / Normalizacion
media = mean(x);
sigma = std(x);
for i=1:m
    xNorm(i,:) = (x(i,:)-media)./sigma;
end

% --------------------- Impresion de datos estandarizados
figure('Name','Datos estandarizados y linea de clasificacion');
figure(2);
hold on;
plot(xNorm(clase0,1), xNorm(clase0, 2), "ok", "MarkerFaceColor", "c"); 
plot(xNorm(clase1,1), xNorm(clase1, 2), "dk", "MarkerFaceColor", "m"); 
title("Calificaciones de alumnos Normalizados");
xlabel("x1 = Parcial 1");
ylabel("x2 = Parcial 2");

% --------------------- Parametros iniciales
X = [ones(m,1), xNorm];
iter = 1;
iterMax = 1200;
beta = 0.4;
a = zeros(n, 1); 

% --------------------- Calcular hipotesis inicial
for i=1:m
    x = X(i,:)';
    z = a'*x;
    h(i,1) = g(z);
end

% --------------------- Impresion de PRIMERA linea de clasificacion
x1 = -2:0.1:2;
x2 = (-a(1)-a(2)*x1)/a(3);
figure(2)
hold on
plot(x1, x2, 'y') 
axis([-2 2 -2 2])

% --------------------- Obtener funcion de costo J
J = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));
convergencia(1) = J;

% --------------------- Entrenamiento 
while iter < iterMax
    for j=1:n
        a(j) = a(j)-beta*(1/m)*sum((h-y).*X(:,j));
    end
    for i=1:m
        x = X(i,:)';
        z = a'*x;
        h(i,1) = g(z);
    end
    J = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));
    convergencia(iter) = J;
    iter = iter + 1;
end

% --------------------- Impresion de linea FINAL de clasificacion
x1 = -2:0.1:2; 
x2 = (-a(1)-a(2)*x1)/a(3);

figure(2)
plot(x1, x2, 'r')
axis([-2 2 -2 2])

% --------------------- Impresion de linea de convergencia
figure('Name','Grafica de convergencia');
figure(3)
plot(convergencia)

% --------------------- Transformamos h
h = h >= 0.5;

% --------------------- Prediccion del modelo
figure('Name','Prediccion del modelo');
figure(4)
hold on
clase0 = find(h == 0);
clase1 = find(h == 1);
plot(X(clase0,2), X(clase0,3), "ok", "MarkerFaceColor", "c");
plot(X(clase1,2), X(clase1,3), "dk", "MarkerFaceColor", "m");
title("Calificaciones")
xlabel("x1: Parcial 1")
ylabel("x2: Parcial 2")

% --------------------- Impresion de ULTIMA linea de clasificacion (para otra grafica)
x1 = -2:0.1:2;
x2 = (-a(1)-a(2)*x1)/a(3);

figure(4)
plot(x1, x2, "r") % dibujar linea de clasificacion
axis([-2 2 -2 2])

% -------- PRUEBAS --------
% --------------------- Datos de prueba e impresion de resultados

datosPrueba = [34.6237 78.0247,
              60.1826 86.3086,
              82.2267 42.7199]; 
i = 1;
while(i<=3)
    datoPrueba = datosPrueba(i,:); 
    
    datoPruebaNorm = (datoPrueba-media)./sigma; 
    datoPruebaNorm = [1 datoPruebaNorm]; 
    z_datoPrueba = a'*datoPruebaNorm'; 
    h_datoPrueba = g(z_datoPrueba); 
    
    fprintf("\nError J = %f     a0 = %f   a1 = %f   a2 = %f \n", J, a(1), a(2), a(3)); 
    
    fprintf("Dato de prueba %d:  x1 = %f  x2 = %f \n", i, datoPrueba(1), datoPrueba(2)); 
    fprintf("Probabilidad de pertenecer a la clase 1:  h = %f   Prediccion: clase %d \n", h_datoPrueba, h_datoPrueba >= 0.5); 
  
    i = i + 1;
end













