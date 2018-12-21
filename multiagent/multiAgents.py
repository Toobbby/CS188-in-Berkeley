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
import random, util,sys

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

        successorGameState = currentGameState.generatePacmanSuccessor(action)#original
        if successorGameState.isWin() :   return float('inf')   #judge the win state
        if successorGameState.isLose() :  return float('-inf')  #judge the lose state

        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "obtain food score"
        newFood = successorGameState.getFood()
        newfoodList = newFood.asList()
        closestFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in newfoodList])
        foodScore = 1.0 / closestFood

        "obtain ghost score"
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        if ghostPositions:
            closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions])
            if closestGhost < 2:      #avoid be eaten
                return float('-inf')
        else:
            return 1   #acting normal when there is no ghost arround
        totalScaredTime = sum(newScaredTimes)#scaredtimes
        score = successorGameState.getScore()
        'define some evaluate functions'
        surroundingFood = self.surroundingFood(newPos, newFood) #bigger is better
        avgFoodDistance = self.avgFoodDistance(newPos, newFood) #less is better
        'after optimize process'
        return 5*(2.0/closestGhost + score + surroundingFood + 10.0/avgFoodDistance) +foodScore/closestGhost + totalScaredTime

    def avgFoodDistance(self, newPos, newFood):
        distances = []
        for x, row in enumerate(newFood):
            for y, column in enumerate(newFood[x]):
                if newFood[x][y]:
                    distances.append(manhattanDistance(newPos, (x,y)))
        avgDistance = sum(distances)/float(len(distances)) if (distances and sum(distances) != 0) else 1
        return avgDistance

    def surroundingFood(self, newPos, newFood):
        count = 0
        for x in range(newPos[0]-2, newPos[0]+3):
            for y in range(newPos[1]-2, newPos[1]+3):
                if (0 <= x and x < len(list(newFood))) and (0 <= y and y < len(list(newFood[1]))) and newFood[x][y]:
                    count += 1
        return count



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
        depth = self.depth
        actions = gameState.getLegalActions(0)
        random.shuffle(actions)
        if len(actions) > 0:
            maximum, bestAction = max((self.minimax(gameState.generateSuccessor(0, action), depth, 1), action) for action in actions)
            # pacman do max, so every actioncalculate the maximum score
            return bestAction
        else:
            return gameState.getLegalActions(0)

    def minimax(self, gameState, depth, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if index == 0:# current agent is Pacman, do max
            actions = gameState.getLegalActions(0)
            if len(actions) > 0:
                return max(self.minimax(gameState.generateSuccessor(0, action), depth, 1) for action in actions)
            else:
                return self.evaluationFunction(gameState)
        else:
            actions = gameState.getLegalActions(index)
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                # nextAgent is Pacman, we decrease the depth
                depth=depth-1

            if len(actions) > 0:
                return min(self.minimax(gameState.generateSuccessor(index, action), depth, nextIndex) for action in actions)
            else:
                return self.evaluationFunction(gameState)


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
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        maxVal = float('-inf')
        for action in gameState.getLegalActions(0):
            val = self.alphaBetaPrun(gameState.generateSuccessor(0, action), depth, alpha, beta, 1)
            if val > maxVal:
                maxVal = val
                bestAction = action
            if maxVal > beta:
                return bestAction
            alpha = max(alpha, maxVal)
        return bestAction

    def alphaBetaPrun(self, gameState, depth, alpha, beta, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        elif index == 0:
            # current agent is Pacman, do Max
            maxVal = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                val = self.alphaBetaPrun(gameState.generateSuccessor(0, action), depth, alpha, beta, 1)
                if val > maxVal:
                    maxVal = val
                if maxVal > beta:
                    return maxVal
                alpha = max(alpha, maxVal)
            return maxVal
        else:
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                # nextAgent is Pacman, decrease the depth
                depth -= 1
            minVal = float('inf')
            actions = gameState.getLegalActions(index)
            for action in actions:
                minVal = min(minVal, self.alphaBetaPrun(gameState.generateSuccessor(index, action), depth, alpha, beta, nextIndex))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
            return minVal

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
        maximum = float('-inf')
        depth = self.depth
        bestAction = gameState.getLegalActions(0)
        for action in gameState.getLegalActions(0):
            value = self.expectimax(gameState.generateSuccessor(0, action), self.depth, 1)
            if value > maximum:
                maximum = value
                bestAction = action
        return bestAction

    def expectimax(self, gameState, depth, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if index == 0:
            # current agent is Pacman, do max
            actions = gameState.getLegalActions(0)
            return max(self.expectimax(gameState.generateSuccessor(0, action), depth, 1) for action in actions)
        else:
            actions = gameState.getLegalActions(index)
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                # nextAgent is Pacman, decrease the depth
                depth=depth-1

            return sum(self.expectimax(gameState.generateSuccessor(index, action), depth, nextIndex) for action in actions)/len(actions)



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I actually optimized the evaluation function in Q1. I consider the score of Food, Ghost: Capsule and Hunter. Finally add them with multiplying weight.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin() :  return float('inf')
    if currentGameState.isLose() :  return float('-inf')
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    capsulePos = currentGameState.getCapsules()
    'initialie'
    weightFood, weightGhost, weightCapsule, weightHunter = 5 , 5 , 5 , 0
    ghostScore, capsuleScore, hunterScore = 0 , 0 , 0

    "obtain food score"
    currentFoodList = currentFood.asList()
    closestFood = min([util.manhattanDistance(currentPos, foodPos) for foodPos in currentFoodList])
    foodScore = 1  / closestFood

    "obtain ghost, capsule, hunting score"
    if GhostStates:
        ghostPositions = [ghostState.getPosition() for ghostState in GhostStates]
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        ghostDistances = [util.manhattanDistance(currentPos, ghostPos) for ghostPos in ghostPositions]

        if sum(ScaredTimes) == 0: # run or catch
            closestGhost = min(ghostDistances)
            ghostCenterPos = ( sum([ghostPos[0] for ghostPos in ghostPositions])/len(GhostStates),\
                               sum([ghostPos[1] for ghostPos in ghostPositions])/len(GhostStates))
            ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
            if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 20, (1  / closestCapsule)
                        weightGhost, ghostScore = 3, (-1 / (ghostCenterDist+1))
                    else:
                        weightGhost, ghostScore = 10, (-1 / (ghostCenterDist+1))
                else:
                    weightGhost, ghostScore = 10, (-1 / (ghostCenterDist+1))
            elif ghostCenterDist >= closestGhost and closestGhost >= 1:
                weightFood *= 2
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 15, (1 / closestCapsule)
                        weightGhost, ghostScore = 3, (-1 / closestGhost)
                    else:
                        ghostScore = -1 / closestGhost
                else:
                    ghostScore = -1 / closestGhost
            elif closestGhost == 0:
                return float('-inf')
            elif closestGhost == 1:
                weightGhost, ghostScore = 15, (-1 / closestGhost)
            else:
                ghostScore = -1 / closestGhost
        else: # hunter mode
            normalGhostDist = []
            closestPrey = float('inf')
            ghostCenterX, ghostCenterY = 0 , 0
            for (index, ghostDist) in enumerate(ghostDistances):
                if ScaredTimes[index] == 0 :
                    normalGhostDist.append(ghostDist)
                    ghostCenterX += ghostPositions[index][0]
                    ghostCenterY += ghostPositions[index][1]
                else:
                    if ghostDist <= ScaredTimes[index] :
                        if ghostDist < closestPrey:
                            closestPrey = ghostDistances[index]
            if normalGhostDist:
                closestGhost = min(normalGhostDist)
                ghostCenterPos = ( ghostCenterX/len(normalGhostDist), ghostCenterY/len(normalGhostDist))
                ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
                if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                    weightGhost, ghostScore = 10, (- 1/(ghostCenterDist+1))
                elif ghostCenterDist >= closestGhost and closestGhost >= 1 :
                    ghostScore = -1 / closestGhost
                elif closestGhost ==0:
                    return float('-inf')
                elif closestGhost ==1:
                    weightGhost, ghostScore = 15, (-1/ closestGhost)
                else:
                    ghostScore = -1/ closestGhost
            weightHunter, hunterScore = 30,(1/closestPrey)

    "evaluation function"
    heuristic = currentGameState.getScore() + weightFood*foodScore + weightGhost*ghostScore + weightCapsule*capsuleScore + weightHunter*hunterScore
    return heuristic

    # Abbreviation
better = betterEvaluationFunction
