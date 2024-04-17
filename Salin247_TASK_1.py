import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation

import heapq

# obstacle type flag
static_only = 0

# Calculate the distance between two points
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Generate the neighbors of a point
def neighbors(point):
    x, y = point
    return [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]


# Check if a point is inside a polygon
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside






# Generate a path from start to goal avoiding static and dynamic obstacles
def generate_path(start, goal, static_obstacles, dynamic_obstacles, time_of_simulation):

    # Function to check if a point is valid (not colliding with any obstacles)
    def is_valid(point):
        for obstacle in static_obstacles: # Check collision with static obstacles
            if point_in_polygon(point, obstacle):
                return False
        for obstacle in dynamic_obstacles: # Check collision with dynamic obstacles
            if distance(point,  obstacle['initial_position'][-1]) <= 1.0:  # Use the current position of dynamic obstacle
                return False
        return True

    def manhattan_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x2 - x1) + abs(y2 - y1)

    # Heuristic function for search based planning
    def heuristic(point):
        return distance(point, goal) + manhattan_distance(point, goal)

    # Initialize open set with start node and heapify it
    open_set = [(heuristic(start), start)]  
    heapq.heapify(open_set)

    # Initialize dictionaries to store parent nodes and G scores for each node
    came_from = {}
    g_score = {start: 0}

    # Initialize time elapsed
    time_elapsed = 0

    while open_set and time_elapsed < time_of_simulation:
        _, current_node = heapq.heappop(open_set)

        # Check if goal is reached
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        # Expand current node by considering its neighbors
        for neighbor in neighbors(current_node):
            tentative_g_score = g_score[current_node] + distance(current_node, neighbor)
             # Check if neighbor is valid and its tentative G score is lower than existing G score
            if is_valid(neighbor) and (neighbor not in g_score or tentative_g_score < g_score[neighbor]):
                # Update parent node and G score for neighbor
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                # Calculate F score and add neighbor to open set
                f_score = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

        # Check if any dynamic obstacles will collide with the path and update the path accordingly
        for obstacle in dynamic_obstacles:
            obstacle_position =  obstacle['initial_position'][-1]

            if distance(current_node, obstacle_position) <= 0.1:
                # Dynamic obstacle in the vicinity, adjust path
                if obstacle_position in g_score and g_score[obstacle_position] > g_score[current_node]:
                    # Adjust cost if obstacle position has been reached before
                    g_score[obstacle_position] = g_score[current_node]
                    # Re-calculate f-score
                    f_score = g_score[obstacle_position] + heuristic(obstacle_position)
                    # Push the adjusted position back to open set
                    heapq.heappush(open_set, (f_score, obstacle_position))

                    # Adjust costs of affected nodes
                    for node in neighbors(obstacle_position):
                        if node in g_score and g_score[node] > g_score[obstacle_position] + distance(obstacle_position, node):
                            g_score[node] = g_score[obstacle_position] + distance(obstacle_position, node)
                            f_score = g_score[node] + heuristic(node)
                            heapq.heappush(open_set, (f_score, node))

        time_elapsed += 1

    return None




# Define the start and goal points
start = (-2, -2)
goal = (8, 6)

# Define the static obstacles as a list of polygons
static_obstacles = [
    [(2, 2), (2, 8), (3, 8), (3, 3), (8, 3), (8, 2)],
    [(6, 6), (7, 6), (7, 7), (6, 7)]
]


# Define the dynamic obstacles as a list of points
dynamic_obstacles = [
    {'initial_position': [
        (10, 1)], "velocity": [random.uniform(-1, 1), random.uniform(-1, 1)]},
    {'initial_position': [
        (2.5, 10)], "velocity": [random.uniform(-1, 1)*.5, random.uniform(-1, 1)*.5]},
    {'initial_position': [
        (5, 5)], "velocity": [random.uniform(-1, 1)*.2, random.uniform(-1, 1)*.2]},
    {'initial_position': [
        (0, 2.5)], "velocity": [random.uniform(-1, 1)*.1, random.uniform(-1, 1)*.1]}
]

# Define functions to plot obstacles
def plot_polygon(polygon, color):
    x, y = zip(*polygon)
    axes.fill(x, y, color=color)


def get_dynamic_obstacle_location(obstacle, frame):
    point = obstacle['initial_position']
    velocity = obstacle['velocity']
    vx, vy = velocity[0], velocity[1]
    x = [i[0] + frame*vx for i in point]
    y = [i[1] + frame*vy for i in point]
    return x, y


fig = plt.figure(figsize=(5, 5))
axes = fig.add_subplot(111)
plt.xlim(-5, 15)
plt.ylim(-5, 15)
plt.xlabel('X')
plt.ylabel('Y')

dynamic_obstacles_location = []

for i, obstacle in enumerate(dynamic_obstacles):
    point, = axes.plot([], [], 'ok', ms=20)
    dynamic_obstacles_location.append(point)



for i, obstacle in enumerate(static_obstacles):
    plot_polygon(obstacle, 'darkgray')


def update_animation(frame):


    # update dynamic obstacles
    my_dynamic_obstacles = []     # <----------- I have added this line
    for i, obstacle in enumerate(dynamic_obstacles):
        x, y = get_dynamic_obstacle_location(obstacle, frame+1)
        dynamic_obstacles_location[i].set_data(x, y)
        my_dynamic_obstacles.append({'initial_position': [(x[0], y[0])]})  # <----------- I have added this line


   
    # TODO: you may compute the path here!

    # Get the points contituting the path
    path = generate_path(start, goal, static_obstacles, my_dynamic_obstacles, time_of_simulation=1000)
 
    # Clear only the lines representing the path
    for line in axes.lines:
        if line.get_color() == 'red':
            line.remove()



    # Plot the path as a red line up to the current frame
    if path:    # <----------- I have added this line (to avoid errors due to None path if at all it exists)
        x = [i[0] for i in path[:frame+1]]
        y = [i[1] for i in path[:frame+1]]
        axes.plot(x, y, color='red')


    # Plot the start and goal points as green and blue circles, respectively
    axes.scatter(start[0], start[1], color='green', s=100)
    axes.scatter(goal[0], goal[1], color='blue', s=100)
    return []


# Create the animation using FuncAnimation
animation = FuncAnimation(fig, update_animation, frames=45, interval=250, blit=True)

# Show the plot
plt.show()
