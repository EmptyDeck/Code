//C++ code


#define LED_PIN LED_BUILTIN
#define LED_ON LOW
#define LED_OFF HIGH

int ledVal;

void setup(){

    Serial.begin(115200);
    Serial.printIn();
    pinMode(LED_PIN,OUTPUT);


}



void loop(){

    while(Serial.avaiable()){
        char c = Serial.read();
        if (c == 'm'){
            int val = Serial.parseInt():
        }
    ledVal = val ? LED_on : LED_OFF;
    digitalWrite(LED_PIN,ledVal)
    }


}