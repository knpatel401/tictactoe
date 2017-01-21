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
    '''
      Print the tic-tac-toe board.  The input is 
      assumed to be a tuple of length 9 with elements 
      0, 1, or -1.
    '''
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
    '''
      Take the current board as input and return the player's whose turn is next.
      This is done simply by checking the sum of the entries on the board.
      The board is assumed to be a tuple of length 9.  The value returned is 1 or -1.  
    '''
    currentStatus = sum(board)
    if currentStatus > 0:
        return -1
    else:
        return 1
    
def possibleMoves(board):
    '''
       Take board as input and return the indices of possible next moves.
    '''
    if evalBoard(board) == 0:
        possible = [i for i,a in enumerate(board) if a==0]
    else:
        possible = []
    return possible
    
def makeRandomMove(board):
    '''
       Make a (valid) random move given the current board.
    '''
    possible = possibleMoves(board)
    move = random.choice(possible)
    board = updateBoard(board, move)
    
    return board, move

# A quick global constant to indicate a clean board.
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

    def possibleMoves(self, board):
        '''
        Take board as input and return the indices of possible next moves.
        '''
        if evalBoard(board) == 0:
            possible = [i for i,a in enumerate(board) if a==0]
        else:
            possible = []
        return possible

    def bestMove(self, board):
        '''
        Given the current board, return the best move available based on the current value function.
        Note: the "best" move is only necessarily the best move if the value function is correct.

        Input is the board (as an 9-tuple)
        Ouput is the updated board and the index of the move.
        '''
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
                
    def logBoard(self, board):
        '''
        Log the current board in the board history.  This is used for MC update 
        of the value function.
        '''
        self._boardHist.append(board)

    def monteCarloUpdate(self):
        '''
        This function does a Monte Carlo update on the game currently logged.
        '''
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

def getMoveFromUser(board):
    done = False
    possible = possibleMoves(board)
    while not done:
      print("                 ")
      printBoard(board)
      print("Valid moves = " + str(possible))
      move = int(raw_input("Please enter your move: "))
      if move in possible:
          done = True
      else:
          print("Invalid move")
    return updateBoard(board,move), move
      
def valueIteration():
    '''Update the value function using value iteration until we converge to the optimal
       value function.  This is essentially using dynamic programming to find the optimal 
       value function.  The bot using this value function will play optimally.'''
    myBot = ticTacToeBot()
    myBot.initValues()
    update = True
    while update:
        update = myBot.iterateValue()

    return myBot
        
def playGame(myBot):
    '''Have the bot (myBot) play a game against itself.'''
    board = newGame
    myBot.logBoard(board)
    while len(possibleMoves(board)) > 0 and evalBoard(board)==0:        
        newBoard, move = myBot.bestMove(board)
        board = newBoard
        myBot.logBoard(board)

    printBoard(board)
    return evalBoard(board)

def runMCiterations(myBot, numIter):
    '''Run Monte Carlo updates of the value function for the specified number of iterations.'''
    myBot._stateVisitCount = {}
    myBot._boardHist = []
    for iter in range(numIter):
        playGame(myBot)
        myBot.monteCarloUpdate()
        
# Simulation of playing against the bot after using value iteration to find optimal policy    
myBot = ticTacToeBot()
myBot.initValues()
myBot = valueIteration()

donePlaying = False
while not donePlaying:

    board = newGame
    gameDone = False
    while not gameDone:
        board, move = getMoveFromUser(board)
        if evalBoard(board) != 0 or len(possibleMoves(board))==0:
            printBoard(board)
            break
        board, move = myBot.bestMove(board)
        if evalBoard(board) != 0 or len(possibleMoves(board))==0:
            break
        
    if evalBoard(board) == 0:
        print("Tie game!")
    if evalBoard(board) == 1:
        print("X wins!")
    if evalBoard(board) == -1:
        print("O wins!")
    
    response = (raw_input("Play again? [n to stop] ")).lower()
    if response[0] == "n":
        break
    
