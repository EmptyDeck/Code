#define LED_PIN LED_BUILTIN
#define LED_OFF 0
#define LED_ON 1
#define LED_BLINK 2

bool onVal = HIGH;
bool offVal = LOW;

int mode;
int onMil = 3000;
int offMil = 500;

void setup()
{
    Serial.begin(115200);
    Serial.println();
    Serial.println("INPUT m0 for led off");
    Serial.println("INPUT m1 for led on");
    Serial.println("INPUT m2 for led blink");

    pinMode(LED_PIN, OUTPUT); // corrected the pin name from LED_FIN to LED_PIN
}

void loop()
{
    while (Serial.available())
    {
        char c = Serial.read();
        if (c == 'm')
        {
            int val = Serial.parseInt(); // corrected the method name from pareseInt to parseInt
            switch (val)
            {
            case 0:
                mode = LED_OFF;
                digitalWrite(LED_PIN, offVal);
                Serial.println("mode = LED_OFF");
                break;
            case 1:
                mode = LED_ON;
                digitalWrite(LED_PIN, onVal);
                Serial.println("mode = LED_ON");
                break;
            case 2:
                mode = LED_BLINK;
                Serial.println("mode = LED_BLINK");
                break;
            default:
                Serial.println("mode Error!");
                continue;
            }
        }
        if (mode == LED_BLINK)
        {
            digitalWrite(LED_PIN, onVal);
            delay(onMil);
            digitalWrite(LED_PIN, offVal);
            delay(offMil);
        }
    }
}
