# tictactoe
This code illustrates some based concepts of reinforcement learning
via the game of tic-tac-toe.  Of course, the game of tic-tac-toe is simple enough
that the optimal strategy can be easily found with a brute force
search, but here the goal is to use this simple game to provide
intution on how reinforcement learning techniques work and can then be
applied to more complicated problems.

The code implements a tic-tac-toe playing bot that learns to play through:
- Value iteration - Use dynamic programming to update the value
  function until convergence.  The resulting bot will play the game
  optimally.
- Monte Carlo update - Play a series of games against itself and
    update the value function based on these games.
	
## TODO:
- Need to add exploration/random component to the MC method so that
  the bot explores the entire space in the self-learning case.
- Need to add example functions to illustrate each method.
- Add function to get moves from user so can play against the bot.
- Add other methods (e.g., temporal difference)
- Add hook for value function approximation
