import itertools
from os import system
from time import sleep
import numpy as np
from numpy import asanyarray as ana
from PIL import Image
import cv2
import pyautogui
# import myOwnLibrary as mol

class blockBlast:


    def __init__(self):
        self.grid = np.zeros((8,8))
        self.left, self.top= 24, 186
        self.right, self.bot = 277, 439
        self.diffX = (self.right - self.left)//9
        self.diffY = (self.bot - self.top)//9
        self.pieces = []
        self.BestMoves = []

    def analyseGrid(self, image: np.ndarray):
        if type(image) != type(np.ndarray):
            image = ana(image)
        blank = ana(Image.open('Screenshots/BlankBoard.png'))[:,:,:3].astype(float)
        image = np.abs(blank - image[self.top:self.bot, self.left:self.right, :3]).sum(axis=2)
        image[image > 20] = 255.0
        
        self.grid = (ana(Image.fromarray(image.astype(np.uint8)).resize((8,8), resample=Image.BOX))/128).astype(int)
        # Image.fromarray(image.astype(np.uint8)).show()
    
    def analysePieces(self, image: np.ndarray):
        if type(image) != type(np.ndarray):
            image = ana(image)[:,:,:3].copy().astype(float)
        image = image[444:585]
        diffX = image.shape[1]//7
        for i in range(0, 6, 2):
            piece = image[:, i*diffX:(i+3)*diffX]
            piece = piece[:,20:-20]
            piece = np.abs(piece[1:,1:] - piece[:-1,:-1])
            piece = piece.sum(axis=2)

            t,b,l,r = self.boundingBox(piece)
            piece = piece[t:b, l:r]
            piece = cv2.resize(piece, (ana(piece.shape[::-1])//14), interpolation=cv2.INTER_AREA)
            piece[piece > 30] = 255.0
            piece = (piece//255)
            self.pieces.append(piece.astype(int))
            # print(self.pieces)
            # Image.fromarray(piece.astype(np.uint8)).show()

    def boundingBox(self, frame):
        '''MY OWN'''
        frame = frame.copy()
        valid = np.where(frame > 30)
        if valid == ():
            return 0,1,0,1
        t = valid[0][0]
        b = valid[0][-1]
        l = np.min(valid[1])
        r = np.max(valid[1])

        return t,b,l,r
    
    def exploreCombinations(self, remainingPieces, tempGrid, sequenceSoFar, allCombinations, pickOrder):
        if not remainingPieces:
            allCombinations.append((pickOrder, sequenceSoFar))
            return

        piece = remainingPieces[0]
        possiblePlacements = self.slidePieces(piece, tempGrid)

        for placement in possiblePlacements:
            newGrid = tempGrid + placement
            newGrid = self.solveBoard(newGrid)
            self.exploreCombinations(
                remainingPieces[1:],
                newGrid,
                sequenceSoFar + [placement],
                allCombinations,
                pickOrder
            )

    def movePieces(self, x, y, width, height):
        for i in self.BestMoves:
            print(f"Moving piece {i[0]}")
            print(f"to position {[int(i) for i in i[1]]}")
            centre = x+width//2 + (width//3.5)*(i[0]-1), y+height//2 + height//3.6

            pyautogui.moveTo(centre)

            yTravel = -8+(self.pieces[i[0]].shape[0]/2 + i[1][0])
            print(yTravel)
            xFactor = ((161-33)/2)*i[0]+33
            xTravel = self.pieces[i[0]].shape[1]/2+i[1][1]

            pyautogui.dragTo(centre[0]+(196*xTravel/8) - xFactor, centre[1]+(196*yTravel/8), duration=0.3, button='left', tween=pyautogui.easeOutQuad)
            # while True:
            #     print(pyautogui.position().x-x, pyautogui.position().y-y)
            #     sleep(1)
            # sleep(0.1)
            # pyautogui.dragTo(centre)
            # sleep(0.5)
            # system.exit(0)


    def findBestMove(self):
        possibleMovePermutations = list(itertools.permutations(self.pieces))

        possibleCombinations = []

        for movePermutation in possibleMovePermutations:
            self.exploreCombinations(
                movePermutation,
                self.grid.copy(),
                [],
                possibleCombinations,
                movePermutation  # <- passed in to track pick order
            )

        combo = possibleCombinations[0]
        for i in combo[0]:
            for j,k in enumerate(self.pieces):
                if i.shape == k.shape:
                    if (i == k).all():
                        i[0,0]=-1
                        oof = [j]
                        z = combo[1][j]
                        z = np.where(z > 0)
                        t = z[0][0]
                        l = np.min(z[1])
                        oof.append([t,l])
                        self.BestMoves.append(oof)

                        break
                    



                

    def solveBoard(self, board):
        x,y = board.sum(axis=0), board.sum(axis=1)
        t = board.copy()
        for i,j in enumerate(x):
            if j == 8:
                t[:,i] =0
        for i,j in enumerate(y):
            if j == 8:
                t[i,:] =0
        return t
    
    def slidePieces(self, piece, board):
        x,y = piece.shape
        piece = np.pad(piece.copy(),((0,8-x),(0,8-y)))
        places = []
        for i in range(9-x):
            for j in range(9-y):
                if not ((board + np.roll(piece, (i,j), axis=(0,1)))//2).any():
                    places.append(np.roll(piece, (i,j), axis=(0,1)))
        return places
        


def tupleOfRegion(x,y,width,height):
    return tuple(int(i) for i in (x,y,width,height))

def waitStillness():
    sleep(3)

# piece = ana([
#     [1,1],
#     [0,0],
#     [1,0]
# ])
# game = blockBlast()
# print(game.slidePieces(piece))