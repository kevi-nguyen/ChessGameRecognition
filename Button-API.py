from fastapi import FastAPI, HTTPException
import requests
import RPi.GPIO as GPIO

app = FastAPI()

class ButtonAPI:
    def __init__(self, button_pin):
        self.button_pin = button_pin
        self.setup_gpio()

    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.button_pin, GPIO.FALLING, callback=self.button_press_callback, bouncetime=300)

    def button_press_callback(self, channel):
        response = requests.post('http://localhost:8080/process_move')
        if response.status_code == 200:
            print("Move made")
        else:
            print(f"Failed to process move: {response.status_code}")


# Initialize ButtonAPI with the GPIO pin number
button_api = ButtonAPI(button_pin=17)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
