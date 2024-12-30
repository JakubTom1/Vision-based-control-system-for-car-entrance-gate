#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>

const char *ssid = "<<subnet_id>>";
const char *password = "<<password>>";

IPAddress local_IP(<<IP_address_of_ESP32>>);        // Static IP address
IPAddress gateway(<<gateway_of_subnet>);           // Gateway
IPAddress subnet(255, 255, 255, 0);          // Mask
IPAddress primaryDNS(8, 8, 8, 8);            // Primary DNS (Google DNS)
IPAddress secondaryDNS(8, 8, 4, 4);          // Alternative DNS (Google DNS)

WebServer server(80); 
const int OPEN_Pin = 14;
const int Signal_LED_Pin = 27;
const int State_LED_Pin = 12;

void signal(int delayTime) { 
  digitalWrite(OPEN_Pin, LOW); // Open/Close gate 
  digitalWrite(Signal_LED_Pin, HIGH); // Turn On LED 
  delay(delayTime); 
  digitalWrite(OPEN_Pin, HIGH); // Signal reset
  digitalWrite(Signal_LED_Pin, LOW); // Turn Off LED 
} 
  
void handleRoot() { 
  // State reading 
  String state = server.arg("state"); 
  String input = Serial.readString(); 
  // State check 
  if (state == "on") { 
    signal(500); // open signal 
    Serial.println("Gate is opening."); 
    server.send(200, "text/plain", "Gate is opening"); 
  } else if (state == "off") { 
    signal(500); //close signal 
    Serial.println("Gate is closing."); 
    server.send(200, "text/plain", "Gate is closing"); 
  } else { Serial.println("Value is not correct"); 
  server.send(400, "text/plain", "Incorrect vallue. Try 'True' or 'False'"); 
  }
} 

void setup() {
  Serial.begin(115200); 
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) { 
    Serial.println("Can't configure static IP!"); 
  } 
  
  // configure 
  pinMode(OPEN_Pin, OUTPUT); 
  digitalWrite(OPEN_Pin, HIGH); // Gate signal

  pinMode(Signal_LED_Pin, OUTPUT); 
  digitalWrite(Signal_LED_Pin, LOW); // LED 

  pinMode(State_LED_Pin, OUTPUT);
  digitalWrite(State_LED_Pin, LOW); // Dioda wyłączona na start

  // Wi-Fi connection 
  int trial = 0;
  WiFi.begin(ssid, password); 
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.println("Connecting by Wi-Fi...");
     trial = trial + 1;
    if (trial == 120){
      ESP.restart();
    }
  } Serial.println("Connected by Wi-Fi"); 
  Serial.print("Adres IP: "); 
  Serial.println(WiFi.localIP()); // Endpoint "/" 
  server.on("/", handleRoot); 

  // Server configuration 
  server.begin(); 
  Serial.println("Server run"); 
} 

void loop() { 
  server.handleClient(); 

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(State_LED_Pin, HIGH); // Zapal diodę LED
  } else {
    digitalWrite(State_LED_Pin, LOW); // Wyłącz diodę LED
    Serial.println("Connection lost. Reconnecting by Wi-Fi...");
    ESP.restart();
  }
}