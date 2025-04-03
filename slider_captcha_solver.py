import os
import time
import pyautogui
import keyboard
from ultralytics import YOLO
import gc
import cv2
import numpy as np
from PIL import Image

class SimpleSliderCaptchaSolver:
    def __init__(self):
        
        #below all postion as per box(x1, y1, x2, y2)

        self.refresh_button_position = (1077, 365) # refresh button x and y middle of the button
        self.slider_button_position = (868, 653) # slider button x and y middle of the button
        self.source_puzzle_position = [822, 430, 921, 530] # first pazzle starting position x no matter what is y.
        self.destination_puzzle_position = [944, 428, 1008, 528] # second puzzle starting position x no matter what is y.
        
        # x1 is puzzle background starting position
        # y1 is puzzle background starting position
        # x2 is puzzle background ending position - x1
        # y2 is puzzle background ending position - y1
        self.puzzle_position = [821, 411, 279, 154] 
        self.extra_poition = 15
        self.extra_poition_if_near = 8 # this variable use when puzzle pices is close
        self.image_path = "puzzle.jpg"
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            model_path = "runs/detect/puzzle_detector/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            
    def solve_captcha(self):
        if not self.model:
            print("Model not loaded. Cannot solve captcha.")
            return False
            
        try:
            pyautogui.moveTo(self.slider_button_position[0], self.slider_button_position[1], duration=0.5)
            pyautogui.mouseDown()
            
            if (self.destination_puzzle_position[0] - self.source_puzzle_position[0]) <= 110:
                move_distance = (self.destination_puzzle_position[0] - self.source_puzzle_position[0]) + self.extra_poition_if_near 
            else:
                move_distance = (self.destination_puzzle_position[0] - self.source_puzzle_position[0]) + self.extra_poition
            pyautogui.moveRel(move_distance - 100, -15, duration=0.5)
            pyautogui.moveRel(100 + 20, 50, duration=0.3)
            pyautogui.moveRel(-20, -20, duration=0.7)
            pyautogui.mouseUp()
            
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Error during captcha solving: {str(e)}")
            return False
    
    def refresh_captcha(self):
        try:
            pyautogui.click(self.refresh_button_position[0], self.refresh_button_position[1])
            time.sleep(1)
        except Exception as e:
            print(f"Error refreshing captcha: {str(e)}")
        
    def update_destination_puzzle_position(self, new_position):
        self.destination_puzzle_position = new_position
        
    def bw_filter(self, img):
        img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        return img_gray

    def get_puzzle_screen(self):
        try:
            screenshot = pyautogui.screenshot(region=tuple(self.puzzle_position))
            img_np = np.array(screenshot)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_gray = self.bw_filter(img_np)
            img_gray_pil = Image.fromarray(img_gray)
            img_gray_pil.save(self.image_path)
            return img_gray_pil
        except Exception as e:
            print(f"Error capturing puzzle screen: {str(e)}")
            return None

    def predict_puzzle_position(self):
        if not self.model:
            print("Model not loaded. Cannot predict position.")
            return []
            
        try:
            results = self.model.predict(
                source=self.image_path,
                conf=0.50,
                save=False,
                show=False,
            )
            
            puzzle_positions = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(float, box)
                    puzzle_positions.append([x1, y1, x2, y2])
            
            puzzle_positions.sort(key=lambda box: box[0])
            return puzzle_positions
        except Exception as e:
            print(f"Error predicting puzzle position: {str(e)}")
            return []
        finally:
            gc.collect()

def main():
    solver = SimpleSliderCaptchaSolver()
    
    print("Simple Slider Captcha Automation")
    print("===============================")
    print("Press 's' to solve captcha")
    print("Press 'r' to refresh captcha")
    print("Press 'p' to print current mouse position")
    print("Press 't' to auto-detect and solve")
    print("Press 'q' to quit")
    
    running = True
    while running:
        try:
            if keyboard.is_pressed('s'):
                solver.solve_captcha()
                time.sleep(0.5)
            
            elif keyboard.is_pressed('r'):
                solver.refresh_captcha()
                time.sleep(0.5)
            
            elif keyboard.is_pressed('p'):
                current_pos = pyautogui.position()
                print(f"Current mouse position: {current_pos}")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('t'):
                print('t pressed')
                if solver.get_puzzle_screen():
                    result = solver.predict_puzzle_position()
                    if len(result) == 2:
                        x1 = result[1][0]
                        solver.update_destination_puzzle_position([solver.puzzle_position[0] + x1 + 4])
                        solver.solve_captcha()
                    else:
                        print("Puzzle position not detected")
                time.sleep(0.5)
            
            elif keyboard.is_pressed('q'):
                running = False
            
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            time.sleep(1)
    
    if solver.model:
        del solver.model
    gc.collect()

if __name__ == "__main__":
    main()


