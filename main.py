import pyautogui
import time
import pygetwindow
import sys
import gameSolver

titles = pygetwindow.getAllTitles()
titles = [i for i in titles if "iPhone" in i]
if not titles:
    print('Application not found')
    sys.exit()

# while True:
#     print(pyautogui.position())
#     time.sleep(1)

pygetwindow.activate()
x,y,width,height = (int(i) for i in pygetwindow.getWindowGeometry(titles[0]))
centre = pyautogui.moveTo(x+width//2,y+height//2)
pyautogui.moveTo(centre)
pyautogui.click()


# classic_button = pyautogui.locateCenterOnScreen('Screenshots/ClassicButton.png', confidence=0.8)
# pyautogui.moveTo(classic_button.x//2, classic_button.y//2)
# print(classic_button, type(classic_button))
# pyautogui.click()

# centreY = y+int(0.7664670658682635*height)
# x1,x2,x3 = x+int(0.22*width),x+int(0.5*width),x+int(0.78*width)
for i in range(10):
    gameSolver.waitStillness()
    state = pyautogui.screenshot(region=gameSolver.tupleOfRegion(x,y,width,height))
    game = gameSolver.blockBlast()
    game.analyseGrid(state)
    game.analysePieces(state)
    game.findBestMove()
    game.movePieces(x,y,width,height)