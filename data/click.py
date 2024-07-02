import pyautogui
import time

# Coordinates where you want to click
click_x = 1053
click_y = 259

# Number of clicks you want to perform
number_of_clicks = 10

for _ in range(number_of_clicks):
    # Move the mouse to the position and click
    pyautogui.click(click_x, click_y)

    # Wait for 1 second before the next click
    time.sleep(1)

print("Finished clicking.")
