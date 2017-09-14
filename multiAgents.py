# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currPos = currentGameState.getPacmanPosition()
        score=successorGameState.getScore()

        "Look at ghost positions"
        ghostPosns = successorGameState.getGhostPositions()
        nextGhostPos = []
        for x,y in ghostPosns:
            nextGhostPos.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])                  #Find possible ghost new positions
        if newPos not in nextGhostPos and newPos not in ghostPosns: score+=1        #Give score a boost if new pacman position is not a ghost position or a possible one

        "Look at food positions"
        foodPos = [(ix,iy) for ix, row in enumerate(newFood) for iy, i in enumerate(row) if i == True]  #Get food coordinates in (x,y) format
        newDist = min([util.manhattanDistance(newPos,pos) for pos in foodPos]) if foodPos else 0        #Get smallest manhattan distance between new position and food
        oldDist = min([util.manhattanDistance(currPos,pos) for pos in foodPos]) if foodPos else 0       #Get smallest manhattan distance between old position and food
        if newDist < oldDist: score+=1      #Give score a boost if new position is closer to the nearest food

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        move = self.minimax(gameState, 0, True, 1)[1]
        return move

    def minimax(self, gameState, depth, isMaximizing, ghostItr):
        numGhosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:  #Terminating conditions
            return (self.evaluationFunction(gameState),None)

        if isMaximizing:    #Maximizing
            bestValue = (float("-inf"), None)
            for move in gameState.getLegalActions(0):
                succState = gameState.generateSuccessor(0, move)
                v = self.minimax(succState, depth, False, 1)        #Do min for all legal successors
                if v[0] > bestValue[0]:    bestValue = (v[0],move)  #Update best move
            return bestValue

        else:       #Minimizing
            bestValue = (float("inf"), None)
            for move in gameState.getLegalActions(ghostItr):
                succState = gameState.generateSuccessor(ghostItr, move)
                "Do pacman layer if all ghost layers are done else do next ghost layer. Increment depth if all ghosts layers are done"
                v = self.minimax(succState, (depth + 1 if ghostItr == numGhosts else depth), (True if ghostItr == numGhosts else False), ghostItr + 1)
                if v[0] < bestValue[0]:    bestValue = (v[0],move)      #Update best move
            return bestValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        move = self.alphaBetaMinimax(gameState, 0, True, 1, float("-inf"), float("inf"))[1]
        return move

    def alphaBetaMinimax(self, gameState, depth, isMaximizing, ghostItr, alpha, beta):
        numGhosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:  #Terminating conditions
            return (self.evaluationFunction(gameState),None)

        if isMaximizing:   #maximizing
            bestValue = (float("-inf"), None)
            for move in gameState.getLegalActions(0):
                succState = gameState.generateSuccessor(0, move)
                v = self.alphaBetaMinimax(succState, depth, False, 1, alpha, beta)       #Do min for all legal successors
                bestValue = (v[0],move) if v[0] > bestValue[0] else bestValue   #Update best move
                if v[0] > beta: return v
                alpha = max(v[0],alpha)                                         #Update alpha
            return bestValue

        else:   #minimizing
            bestValue = (float("inf"), None)
            for move in gameState.getLegalActions(ghostItr):
                succState = gameState.generateSuccessor(ghostItr, move)
                "Do pacman layer if all ghost layers are done else do next ghost layer. Increment depth if all ghosts layers are done"
                v = self.alphaBetaMinimax(succState, (depth + 1 if ghostItr == numGhosts else depth), (True if ghostItr == numGhosts else False), ghostItr + 1, alpha, beta)
                bestValue = (v[0],move) if v[0] < bestValue[0] else bestValue       #Update best move
                if v[0] < alpha:    return v
                beta = min(v[0],beta)                                               #Update beta
            return bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        move = self.expectimax(gameState, 0, True, 1)[1]
        return move

    def expectimax(self, gameState, depth, isMaximizing, ghostItr):
        numGhosts = gameState.getNumAgents() - 1
        if gameState.isWin() or gameState.isLose() or depth == self.depth:  #Terminating conditions
            return (self.evaluationFunction(gameState),None)

        if isMaximizing:    #Maximizing
            bestValue = (float("-inf"), None)
            for move in gameState.getLegalActions(0):
                succState = gameState.generateSuccessor(0, move)
                v = self.expectimax(succState, depth, False, 1)        #Do min for all legal successors
                if v[0] > bestValue[0]:    bestValue = (v[0],move)     #Update best move
            return bestValue

        else:       #Minimizing
            values = []
            for move in gameState.getLegalActions(ghostItr):
                succState = gameState.generateSuccessor(ghostItr, move)
                "Do pacman layer if all ghost layers are done else do next ghost layer. Add all results to values list. Increment depth if all ghosts layers are done"
                values.extend([self.expectimax(succState, (depth + 1 if ghostItr == numGhosts else depth), (True if ghostItr == numGhosts else False), ghostItr + 1)])
            weight = (1.0/len(gameState.getLegalActions(ghostItr)))     #Assign uniform weight to all actions
            expectation = sum(weight*value[0] for value in values)      #Find expectations by multiplying weight with values and taking sum
            return (expectation,None)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:  Four parameters are looked at to evaluate the state:
                    (1) Distances with food positions, (2) Distance with ghosts, (3) Distance with capsules and (4) Scared time of ghosts

                    Distances with food positions:
                        Get distance to nearest food. Take reciprocal of this value, means being nearer to food is better. Assigned a factor of 5 to this value.
                    Distance with ghosts:
                        Get sum of distances with all ghosts. Assigned a factor of 2 to this value. Farther the distance, the better.
                    Distance with capsules:
                        Same as the case of food, try and get nearer to capsules.
                    Scared time of ghosts:
                        Get minimum scared time of ghosts. Get distance to the nearest ghost.
                        If the number of moves to get to a scared ghost is atleast twice (to be safe, accounting for the movement of the ghost) the distance to the nearest one, assign a factor of 4 to those moves. Additionally, check if all the ghosts are scared before going for the nearest one, as it could be a respawned ghost (not scared anymore).
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPosns = currentGameState.getGhostPositions()
    capsules = currentGameState.getCapsules()

    "Get closer to food"
    foodPos = [(ix,iy) for ix, row in enumerate(foodGrid) for iy, i in enumerate(row) if i == True]         #Get food coordinates in (x,y) format
    foodMinDist = min([util.manhattanDistance(pacmanPos,food) for food in foodPos]) if foodPos else 0       #Get distance to nearest food
    foodDistEval = 1/foodMinDist if foodPos else 0                                                          #Nearer the food, better the evaluation

    "Keep distance with ghosts"
    ghostDist = sum([util.manhattanDistance(pacmanPos,ghostPos) for ghostPos in ghostPosns])                #Add up distance to all ghosts, farther the better

    "Get to a capsule"
    capDist = min([util.manhattanDistance(pacmanPos,capsule) for capsule in capsules]) if capsules else 0   #Get distance to nearest capsule
    capEval = 1/capDist if capsules else 0                                                                  #Nearer the capsule, better the evaluation

    "Chasing ghosts"
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]                                                #Get remaining scared time
    ghostMinDist = min([util.manhattanDistance(pacmanPos,ghostPos) for ghostPos in ghostPosns]) if ghostPosns else 1    #Look at nearest ghost
    pelletEval = 0
    if min(scaredTimes) >= 2*ghostMinDist and len(scaredTimes)==len(ghostPosns):
        pelletEval = min(scaredTimes) if ghostPosns else 0       #If the nearest ghost can be caught up comfortably within remaining time, and if it is scared, get closer to it

    return currentGameState.getScore() + 2*ghostDist + 5*capEval + 5*foodDistEval + 4*pelletEval

# Abbreviation
better = betterEvaluationFunction

