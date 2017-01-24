import random

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
    

# A quick global constant to indicate a clean board.
newGame = (0,0,0,0,0,0,0,0,0)
USE_VALUE_ITERATION, USE_MONTE_CARLO, USE_Q_LEARNING, USER_INPUT = range(4)

class ticTacToeBot(object):
    '''Bot to learn tic tac toe using reinforcement learning'''

    def __init__(self, initBoard = newGame):
        self._mode = USE_MONTE_CARLO
        self._stepSize = 0
        self._epsilon = 0
        self._boardHist = []
        self._boardValues = {}
        self._stateActionValues = {}
        self._stateVisitCount = {}
        self.discountRate = 1
        self.currentBoard = initBoard

    def setStepSize(self, stepSize):
        self._stepSize = stepSize

    def setEpsilon(self, epsilon):
        self._epsilon = epsilon

    def setMode(self, mode):
        self._mode = mode
        
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
                    self._stateActionValues[(board,move)] = reward
                    if reward == 0:
                        nextMoves[newBoard] = self.possibleMoves(newBoard)
                    else:
                        self._stateActionValues[(newBoard)] = 0  # see if we need this
            moves = nextMoves

        
    def possibleMoves(self, board):
        '''
        Take board as input and return the indices of possible next moves.
        '''
        if evalBoard(board) == 0:
            possible = [i for i,a in enumerate(board) if a==0]
        else:
            possible = []
        return possible

    def getMove(self, board ):
        '''
        Given the current board, return the move based on the:
            - current value function (or alternately state-value function)
            - epsilon 
        Note: the "best" move is only necessarily the best move if the value (state-value) function is correct and epsilon is 0

        Input is the board (as an 9-tuple)
        Ouput is the updated board and the index of the move.
        '''
        if self._mode == USER_INPUT:
            return self.getMoveFromUser(board)

        player = nextPlayer(board)
        possible = self.possibleMoves(board)
        bestMove = -1
        bestValue = 0
        randVal = random.uniform(0,1)
        if randVal > self._epsilon:
          for move in possible:
              if self._mode == USE_MONTE_CARLO or self._mode == USE_VALUE_ITERATION:
                  value = player * self._boardValues[ updateBoard( board, move) ]
              else:
                  value = player * self._stateActionValues[(board,move)]
#                  print move,value
              if bestMove == -1 or value > bestValue:
                  bestValue = value
                  bestMove = move
        else:
          bestMove = random.choice(possible)
#          print("random")
            
        return updateBoard( board, bestMove), bestMove

    def getMoveFromUser(self, board):
      done = False
      possible = possibleMoves(board)
      while not done:
        print("                 ")
        printBoard(board)
        print("Valid moves = " + str(possible))
        rawMove = raw_input("Please enter your move: ")
        if rawMove.isdigit():
            move = int(rawMove)
        else:
            print("Entry not a positive integer -- choosing random move")
            move = random.choice(possible)
        if move in possible:
            done = True
        else:
            print("Invalid move")
      return updateBoard(board,move), move
      

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
        if self._mode != USE_MONTE_CARLO:
            return

        discountedReward = evalBoard(self._boardHist[-1])
        for board in reversed(self._boardHist):
            #print board
            numStateVisits = self._stateVisitCount.get(board,0)
            #print numStateVisits
            self._boardValues[board] = 1.0 * numStateVisits / (numStateVisits + 1) * self._boardValues[board] + 1.0 * discountedReward / (numStateVisits + 1)
            self._stateVisitCount[board] = numStateVisits + 1
            discountedReward = self.discountRate * discountedReward
        self._boardHist = []

    def QUpdate(self, board, move):
        '''
        This function does a Temporal difference (TD) update given a board and move.
        '''
        if self._mode != USE_Q_LEARNING:
            return
        newBoard = updateBoard(board, move)
        reward = evalBoard(newBoard)
        nextMoves = possibleMoves(newBoard)

        if len(nextMoves) > 0:
            player = nextPlayer(newBoard)
            maxNextMoveQval = player * max([player * self._stateActionValues[(newBoard,nextMove)] for nextMove in nextMoves])
        else:
            maxNextMoveQval = 0
        #print board,move
        #print self._stateActionValues[(board,move)]
        self._stateActionValues[(board,move)] = self._stateActionValues[(board,move)] + self._stepSize * (reward + self.discountRate * maxNextMoveQval - self._stateActionValues[(board,move)])
        #print self._stateActionValues[(board,move)]
        

            
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
    '''Update the value function using value iteration until we converge to the optimal
       value function.  This is essentially using dynamic programming to find the optimal 
       value function.  The bot using this value function will play optimally.'''
    myBot = ticTacToeBot()
    myBot.initValues()
    update = True
    while update:
        update = myBot.iterateValue()

    return myBot
        

def runMCiterations(myBot, numIter, epsilon=0):
    '''Run Monte Carlo updates of the value function for the specified number of iterations.'''
    myBot._stateVisitCount = {}
    myBot._boardHist = []
    player = 1
    for iter in range(numIter):
        playGame(myBot, epsilon)
        myBot.monteCarloUpdate()
        myBot._boardHist = []
        player = player * -1
    return myBot

def runQiterations(myBot, numIter, epsilon=0):
    '''Run Q-learning updates of the state-value function for the specified number of games.'''
    player = 1
    for iter in range(numIter):
        playGame(myBot, epsilon)
        myBot._boardHist = []
        player = player * -1
    return myBot
def playGame(myBot, epsilon=0):
    '''Have the bot (myBot) play a game against itself.'''
    board = newGame
    myBot.logBoard(board)
    while len(possibleMoves(board)) > 0 and evalBoard(board)==0:
        randVal = random.uniform(0,1)
        if randVal > epsilon:
            newBoard, move = myBot.getMove(board)
        else:
            move = random.choice(possibleMoves(board))
            newBoard = updateBoard(board, move)
        myBot.QUpdate(board, move) 
            
        board = newBoard
        myBot.logBoard(board)

    #printBoard(board)
    return evalBoard(board)

    
def getOptimalBot():
    '''Update the value function using value iteration until we converge to the optimal
       value function.  This is essentially using dynamic programming to find the optimal 
       value function.  The bot using this value function will play optimally.'''

    myBot = ticTacToeBot()
    myBot.initValues()
    myBot.setMode(USE_VALUE_ITERATION)
    update = True
    while update:
        update = myBot.iterateValue()

    return myBot

def getMonteCarloBot():
    myBot = ticTacToeBot()
    myBot.initValues()
    myBot.setMode(USE_MONTE_CARLO)

    return myBot

def getQlearningBot():
    myBot = ticTacToeBot()
    myBot.initValues()
    myBot.setMode(USE_Q_LEARNING)

    return myBot

def getUserBot():
    myBot = ticTacToeBot()
    myBot.setMode(USER_INPUT)

    return myBot


def BotVsBot(myBot1, myBot2, player=1, verbose=False):

    if player != 1 and player != -1:
        raise ValueError("Invalid player values.  Should be 1 or -1")
    board = newGame
    myBot1.logBoard(board)
    myBot2.logBoard(board)
    while len(possibleMoves(board)) > 0 and evalBoard(board)==0:
        if player == 1:
            newBoard, move = myBot1.getMove(board)
        else:
            newBoard, move = myBot2.getMove(board)
        if verbose:
            printBoard(newBoard)
        myBot1.QUpdate(board, move)
        myBot2.QUpdate(board, move)
        board = newBoard
        myBot1.logBoard(board)
        myBot2.logBoard(board)
        player = player * -1
    myBot1.monteCarloUpdate()
    myBot2.monteCarloUpdate()

    return evalBoard(board)

def playGames(myBot1, myBot2, iterations = -1):
    
  numGames = numTies = bot1Win = bot2Win = 0
  
  donePlaying = False
  while not donePlaying:

      #player = random.choice([1,-1])
      player = 1
      result = player * BotVsBot(myBot1, myBot2, player)
      numGames = numGames + 1
      
      if result == 0:
          numTies = numTies + 1
      if result == 1:
          bot1Win = bot1Win + 1
      if result == -1:
          bot2Win = bot2Win + 1

      if iterations == -1:
          response = (raw_input("Play again? [n to stop] ")).lower()
          if len(response) > 0 and response[0] == "n":
              break
          
      if numGames == iterations:
          donePlaying = True
          
  return [1.0 * val / numGames for val in [numTies, bot1Win, bot2Win]]


def main():

    # --------------
    # Generate bots
    # --------------

    # Bot that plays optimally
    # This is trained via value iteration (dynamic programming)
    optBot = getOptimalBot()

    # Generate user input "bot"
    user = getUserBot()

    # Generate bot trained via Monte Carlo 
    mcBot = getMonteCarloBot()
    mcBot.setEpsilon(.1)

    qBot = getQlearningBot()
    qBot.setEpsilon(.1)
    qBot.setStepSize(1)

    # ---------------------------------------
    # Train MC and Q bots against optimal bot
    # ---------------------------------------
    print("Training Monte Carlo Bot")
    print(playGames(optBot, mcBot, iterations = 100000))
    print("Training Q-learning Bot")
    print(playGames(optBot, qBot, iterations = 100000))

    # make bots greedy 
    mcBot.setEpsilon(0)
    qBot.setEpsilon(0)

    # Evaluate against optimal bot
    # Note just a simple sanity check -- outcome doesn't mean MC and Q bots are optimal
    # (particularly since optBot is deterministic)
    print("Testing Monte Carlo Bot")   
    print(playGames(optBot, mcBot, iterations = 1000))
    print("Testing Q-learning Bot")   
    print(playGames(optBot, qBot, iterations = 1000))

    
try:
    if __IPYTHON__:
        pass
except NameError:
    if __name__ == '__main__':
        main()
