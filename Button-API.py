import time

import serial
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Initialize serial connection
port = '/dev/tty.usbserial-AQ027FTG'
baudrate = 9600
ser = serial.Serial(port, baudrate, timeout=1)

# Variables to track button status
button_was_pressed = False
button_is_pressed = False


@app.get("/button_status")
def get_button_status():
    """
    Endpoint to get the current status of the USB button.
    Reflects "pressed" once upon initial press and "not pressed" otherwise.
    """
    global button_was_pressed, button_is_pressed

    # Check for data from the serial port
    if ser.in_waiting > 0:
        data = ser.read(ser.in_waiting)
        button_is_pressed = True
        if not button_was_pressed:
            button_was_pressed = True
            return {"status": "pressed"}
    else:
        button_is_pressed = False
        if button_was_pressed:
            button_was_pressed = False
            return {"status": "not pressed"}

    return {"status": "not pressed"}


@app.get("/wait-for-button-press")
def wait_for_button_press():
    """
    Endpoint to wait for the button to be pressed and then return the status.
    """
    start_time = time.time()
    timeout = 60

    while True:
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                return {"status": "pressed"}
            # Check for timeout
            if time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Request timed out")

            time.sleep(0.1)

        except serial.SerialException as e:
            raise HTTPException(status_code=500, detail=f"Serial error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8084)
