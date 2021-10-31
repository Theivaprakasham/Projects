// Initialzing two variables to store the state of the buttons
int buttonState1, buttonState2;      // variable for reading the pushbutton status

void setup() {
  //start serial connection
  Serial.begin(9600);

  //Setting PinMode
  DDRB = B00001100; // Setting button1 = D8(Input) ;  button2 = D9(Input);  RedLed =  D10(Output); WhiteLed =  D11(Output);
}

void loop() {
  
  // read the state of the pushbutton value:
   buttonState1 = PINB & B00000001;  //performing AND operation to PINB to extract D8 value
   buttonState2 = PINB & B00000010;  //performing AND operation to PINB to extract D9 value


  // Checking the status of Button1
  if (buttonState1 == B00000001) { PORTB = PORTB | B00000100;  // turn on Red Led(D10)
    Serial.println("Red Button is pushed ON. Pull-Down Resistor switches ON the RED LED");  
  } else {   
    PORTB = PORTB & B11111011;  // turn RED LED off:
  }

   // Checking the status of Button2
  if (buttonState2 == B00000010) { PORTB = PORTB | B00001000;  // turn ON White Led(D11)
  } else {
    PORTB = PORTB & B11110111;  // turn WHITE LED off:
    Serial.println("White Button is pushed ON. Pull-UP Resistor switches OFF the White LED");
  }
}
