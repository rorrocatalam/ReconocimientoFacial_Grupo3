const int blueLedPin = 12;  // Pin del LED azul
const int yellowLedPin = 11;  // Pin del LED amarillo
const int redLedPin = 10;  // Pin del LED rojo
unsigned long ledOnDuration = 3000;  // Duración en milisegundos que el LED permanecerá encendido

void setup() {
  pinMode(blueLedPin, OUTPUT);
  pinMode(yellowLedPin, OUTPUT);
  pinMode(redLedPin, OUTPUT);
  Serial.begin(9600);  // Inicia la comunicación serial a 9600 baudios

  digitalWrite(yellowLedPin, HIGH);  // Enciende el LED amarillo al inicio
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');  // Lee el mensaje hasta el final de la línea

    if (command == "BLUE_ON") {
      digitalWrite(yellowLedPin, LOW);  // Apaga el LED amarillo
      digitalWrite(blueLedPin, HIGH);  // Enciende el LED azul
      delay(ledOnDuration);  // Espera durante la duración especificada
      digitalWrite(blueLedPin, LOW);  // Apaga el LED azul
      digitalWrite(yellowLedPin, HIGH);  // Vuelve a encender el LED amarillo
      Serial.println("LED azul encendido y luego apagado");
    } else if (command == "RED_ON") {
      digitalWrite(yellowLedPin, LOW);  // Apaga el LED amarillo
      digitalWrite(redLedPin, HIGH);  // Enciende el LED rojo
      delay(ledOnDuration);  // Espera durante la duración especificada
      digitalWrite(redLedPin, LOW);  // Apaga el LED rojo
      digitalWrite(yellowLedPin, HIGH);  // Vuelve a encender el LED amarillo
      Serial.println("LED rojo encendido y luego apagado");
    }
  }
}
