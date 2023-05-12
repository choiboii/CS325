# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
# A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - ANDREW CHOI, AJCHOI5

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        score = successorGameState.getScore()
        ghostPositions = successorGameState.getGhostPositions()
        for food in newFood.asList():
            score += 1 / manhattanDistance(newPos, food)
        for ghostPosition in ghostPositions:
            if ghostPosition == newPos and 0 in newScaredTimes:
                return -1
            else:
                score += 1
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # max function; calls min function when not explored full depth yet
        def maximum(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            nextStates = []
            for action in state.getLegalActions(0):
                if action == 'STOP':
                    continue
                else:
                    nextStates.append(state.generateSuccessor(0, action))
            temp = float("-inf")
            for nState in nextStates:
                temp = max(temp, minimum(nState, depth, 1))
            return temp
                
        # min function; calls either max function when not explored full depth or min function if there are multiple ghosts
        def minimum(state, depth, agent_idx):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            nextStates = []
            for action in state.getLegalActions(agent_idx):
                if action == 'STOP':
                    continue
                else:
                    nextStates.append(state.generateSuccessor(agent_idx, action))
            temp = float("inf")
            for nState in nextStates:
                if agent_idx < state.getNumAgents() - 1:
                    temp = min(temp, minimum(nState, depth, agent_idx + 1))
                else:
                    temp = min(temp, maximum(nState, depth + 1))
            return temp

        val = float("-inf")

        # initial max
        for action in gameState.getLegalActions(0):
          temp = minimum(gameState.generateSuccessor(0,action), 0, 1)
          if temp > val:
            val = temp
            move = action
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # similar to minimax alogrithm, except alpha-beta pruning allows ability to skip exploring other successor moves
        def maximum(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            temp = float("-inf")
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                temp = max(temp, minimum(nextState, depth, 1, alpha, beta))
                if temp > beta:
                    return temp
                alpha = max(alpha, temp)
            return temp   
        
        def minimum(state, depth, agent_idx, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            temp = float("inf")
            for action in state.getLegalActions(agent_idx):
                nextState = state.generateSuccessor(agent_idx, action)
                if agent_idx < state.getNumAgents() - 1:
                    temp = min(temp, minimum(nextState, depth, agent_idx + 1, alpha, beta))
                else:
                    temp = min(temp, maximum(nextState, depth + 1, alpha, beta))
                if temp < alpha:
                    return temp
                beta = min(beta, temp)
            return temp

        # initialize alpha, beta, and val values; val = -inf, as we start with max algorithm first
        alpha = float("-inf")
        beta = float("inf")
        val = float("-inf")

        # initial max
        for action in gameState.getLegalActions(0):
          temp = minimum(gameState.generateSuccessor(0,action), 0, 1, alpha, beta)
          if temp > val:
            val = temp
            move = action
            alpha = max(temp, alpha)
        return move


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
        # max function; calls min function when not explored full depth yet
        def maximum(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            nextStates = []
            for action in state.getLegalActions(0):
                if action == 'STOP':
                    continue
                else:
                    nextStates.append(state.generateSuccessor(0, action))
            temp = float("-inf")
            for nState in nextStates:
                temp = max(temp, prob(nState, depth, 1))
            return temp
                
        # prob function; calls either max function when not explored full depth or prob function again with multiple ghosts
        def prob(state, depth, agent_idx):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            temp = 0
            if agent_idx < state.getNumAgents() - 1:
                for action in state.getLegalActions(agent_idx):
                    nextState = state.generateSuccessor(agent_idx, action)
                    temp += prob(nextState, depth, agent_idx + 1)
                temp /= len(state.getLegalActions(agent_idx))  # equal probability
            else:
                for action in state.getLegalActions(agent_idx):
                    nextState = state.generateSuccessor(agent_idx, action)
                    temp += maximum(nextState, depth + 1)   
            return temp

        val = float("-inf")
        # initial max
        for action in gameState.getLegalActions(0):
          temp = prob(gameState.generateSuccessor(0,action), 0, 1)
          if temp > val:
            val = temp
            move = action
        return move

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    For our evaulation function, there are three main priorities that the pacman follows:
    1. Eat all of the food. Pretty straightforward; this is determined through finding the distances between the currPos and 
        all positions of food, and placing an emphasis on minimizing this value (aka reciprocal of the distance).
    2. Avoid the ghosts. This is determined by finding the reciprocal distance again, but this time, putting a negative value
        whenever the pacman ever gets too close. That way, we can still usually find the most optimal route without being too 
        afraid of the ghosts.
    3. Eat the capsules. Along with an added 100 point bonus, the pacman becomes invincible for a short period of time, and they 
        can also eat the ghosts for an extra point bonus. However, the game ends whenever we eat all of the food, so we will only
        put a partial priority on this objective.

    We are able to achieve this through keeping track of many variables in the currentGameState, then evaluating our score based
    on our current score, the distance to the nearest food, the distance to the nearest ghost, and the number of capsules still
    present.
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currCapsules = currentGameState.getCapsules()
    currGhostStates = currentGameState.getGhostStates()
    currGhostPositions = currentGameState.getGhostPositions()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    score = currentGameState.getScore()
    if currPos in currGhostPositions and 0 in currScaredTimes:
        return -1
    if currPos in currFood:
        return 1
    
    foodDistances = []
    ghostDistances = []
    for food in currFood.asList():
        foodDistances.append(manhattanDistance(currPos, food))
    for ghostPos in currGhostPositions:
        ghostDistances.append(manhattanDistance(currPos, ghostPos))
    
    # if the number of capsules is less than 2, add 100 to the score. Otherwise emphasize grabbing the capsule until there is only 1 left.
    if len(currCapsules) < 2:
        score += 100

    # closest food
    if not foodDistances:
        minFood = 0
    else:
        minFood = 1/min(foodDistances)

    # nearest ghost
    if not ghostDistances:
        minGhost = 0
    else:
        minGhost = 1/min(ghostDistances)
    # emphasize going towards the food rather than the ghost, given by the score total
    score += minFood * 10 + minGhost
        
    return score

# Abbreviation
better = betterEvaluationFunction
