from numpy import matrix
import random
import numpy as np

def evalBoard(board):
    '''
      Take a tic-tac-toe board as input and return 
       -1 - if there's a win for Os
        1 - if there's a win for Xs
        0 - if there's no win

      The board is represented as a tuple of length 9 with elements 0, 1 or -1. 
      The indices of the entries correspond to the following locations on the board:
       0 | 1 | 2
       ----------
       3 | 4 | 5
       ----------
       6 | 7 | 8

    '''
    rows = [board[0] + board[1] + board[2],
            board[3] + board[4] + board[5],
            board[6] + board[7] + board[8]]
    cols = [board[0] + board[3] + board[6],
            board[1] + board[4] + board[7],
            board[2] + board[5] + board[8]]
    diag1 = board[0] + board[4] + board[8]
    diag2 = board[2] + board[4] + board[6]

    complete = cols + rows + [diag1, diag2]
    Owin = any([elem==3 for elem in complete])
    Xwin = any([elem==-3 for elem in complete])
    if Owin and Xwin:
        raise ValueError('Invalid board -- wins for both Os and Xs')
    result = Owin*1 + Xwin*-1
    return result

def printBoard(board):
    map = [' ','X','O']

    print('     ' + map[board[0]] + '|' + map[board[1]] + '|' + map[board[2]])
    print('     ' + '-----')
    print('     ' + map[board[3]] + '|' + map[board[4]] + '|' + map[board[5]])
    print('     ' + '-----')
    print('     ' + map[board[6]] + '|' + map[board[7]] + '|' + map[board[8]])

def updateBoard(board, move):
    ''' 
        Given the current board and next move,
        return the updated board.
    '''
    newBoard = list(board)
    newBoard[move] = nextPlayer(board)
    return tuple(newBoard)

def nextPlayer(board):
    currentStatus = sum(board)
    if currentStatus > 0:
        return -1
    else:
        return 1
    
def possibleMoves(board):
    if evalBoard(board) == 0:
        possible = [i for i,a in enumerate(board) if a==0]
    else:
        possible = []
    return possible
    
def makeRandomMove(board):
    '''
    '''
    possible = possibleMoves(board)
    move = random.choice(possible)
    board = updateBoard(board, move)
    
    return board, move

newGame = (0,0,0,0,0,0,0,0,0)

class ticTacToeBot(object):
    '''Bot to learn tic tac toe using reinforcement learning'''

    def __init__(self, initBoard = newGame):
        self._boardHist = []
        self._boardValues = {}
        self._policy = {}
        self._stateVisitCount = {}
        self.discountRate = 1
        self.currentBoard = initBoard

    def board2State(self,board):
        return tuple(np.reshape(board,9).tolist()[0])

    def possibleMoves(self, board):
        if evalBoard(board) == 0:
            possible = [i for i,a in enumerate(board) if a==0]
        else:
            possible = []
        return possible

    def bestMove(self, board):
        player = nextPlayer(board)
        possible = self.possibleMoves(board)
        bestMove = -1
        bestValue = 0
        for move in possible:
            value = player * self._boardValues[ updateBoard( board, move) ]
            if bestMove == -1 or value > bestValue:
                bestValue = value
                bestMove = move
        return updateBoard( board, bestMove), bestMove
                
        
    def makeMove(self, board):
        newBoard, move = self.bestMove(board)
    
        return newBoard,move

    def logBoard(self, board):
        self._boardHist.append(board)

    def monteCarloUpdate(self):
        discountedReward = evalBoard(self._boardHist[-1])
        for board in reversed(self._boardHist[-1]):
            numStateVisits = self._stateVisitCount.get(board,0)
            self._boardValues[board] = 1.0 * numStateVisits / (numStateVisits + 1) * self._boardValues[board] + 1.0 * discountedReward / (numStateVisits + 1)
            self._stateVisitCount[board] = numStateVisits + 1
            discountedReward = self.discountRate * discountedReward
        
    def initValues(self):
        ''' Fill the _boardValues dictionary with all possible valid tic-tac-toe boards 
            and initialize the values with the rewards (i.e., 0 for no win, 1 for player "1" 
            win, and -1 for player "-1" win.
        '''

        self._boardValues[newGame] = evalBoard(newGame)
        possible = self.possibleMoves(newGame)
        moves = {newGame : possible}

        while len(moves)>0:
            nextMoves = {}
            for board,posMoves in moves.items():
#                print board,posMoves
                for move in posMoves:
                    newBoard = updateBoard(board, move)
                    reward = evalBoard(newBoard)
                    self._boardValues[newBoard] = reward
                    if reward == 0:
                        nextMoves[newBoard] = self.possibleMoves(newBoard)
            moves = nextMoves

    def updatePolicy(self):
        '''Update policy based optimal policy given the current values assigned to each state'''

        for board,value in self._boardValues.items():
            _,self._policy[board] = self.bestMove(board)
            
    def iterateValue(self):
        ''' Async value iteration'''

        anyUpdate = False
        for board,value in self._boardValues.items():
            possible = possibleMoves(board)
            player = nextPlayer(board)
            valuesInNextStates = []
#            print board,possible
            if len(possible) > 0 :
              for move in possible:
                  valuesInNextStates.append(player * self._boardValues[updateBoard(board, move)])
              valueOfBestMove = player * max(valuesInNextStates)
                
              newBoardValue = evalBoard(board) + self.discountRate * valueOfBestMove
              if newBoardValue != self._boardValues[board]:
                  anyUpdate = True
                  self._boardValues[board] = newBoardValue
                  
        return anyUpdate
    
def valueIteration():
    myBot = ticTacToeBot()
    myBot.initValues()
    update = True
    while update:
        update = myBot.iterateValue()

    return myBot
        
def playGame(myBot):
    board = newGame
    myBot.logBoard(board)
    while len(possibleMoves(board)) > 0 and evalBoard(board)==0:
        # temp
        #printBoard(board)
        #posMoves = possibleMoves(board)
        #for move in posMoves:
        #    print(move,myBot._boardValues[updateBoard(board,move)])

        
        newBoard, move = myBot.bestMove(board)
        board = newBoard
        myBot.logBoard(board)

    printBoard(board)
    return evalBoard(board)

def runMCiterations(myBot, numIter):
    myBot._stateVisitCount = {}
    myBot._boardHist = []
    for iter in range(numIter):
        playGame(myBot)
        myBot.monteCarloUpdate()
        
    
myBot = ticTacToeBot()
myBot.initValues()
playGame(myBot)

myBot = valueIteration()
playGame(myBot)

