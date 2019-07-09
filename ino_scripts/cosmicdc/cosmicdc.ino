const int PHOTODIODES[] = {A0, A1, A2, A3};
const int sunThreshold = 100;


long lastSunrise;
long lastSunset;
bool isDay;


void setup() {
  for (int i = 0; i < sizeof(PHOTODIODES); i ++)
    pinMode(PHOTODIODES[i], INPUT);

  lastSunrise = 0;
  lastSunset = 0;
}


void loop() {
  checkForSunwend();
}


void checkForSunwend() {
  bool wasDay = isDay;
  isDay = false;
  for (int i = 0; i < sizeof(PHOTODIODES); i ++)
    if (analogRead(PHOTODIODES[i]) > sunThreshold)
      isDay = true;
      
  if (isDay && !wasDay)
    lastSunrise = millis();
  if (!isDay && wasDay)
    lastSunset = millis();
}
