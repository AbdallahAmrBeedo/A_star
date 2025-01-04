import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

def neighbors(map: np.ndarray, point: tuple) -> list:
    '''
    Returns the neighbors of a given point in the map.
    
    Args:
        map: The map of the maze.
        point: The point to find the neighbors of.

    Returns:
        A list of neighbors of the given point.
    '''
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            x, y = point[0] + i, point[1] + j
            if x >= 5 and x < map.shape[0]-2 and y >= 0 and y < map.shape[1] and map[x, y] == 255 and (not nearest_obs(map, (x, y))):
                neighbors.append((x, y))
    return neighbors

def nearest_obs(map: np.ndarray, point: tuple) -> int:
    '''
    Returns the distance to the nearest obstacle from the given point.
    
    Args:
        map: The map of the maze.
        point: The point to find the distance to the nearest obstacle from.

    Returns:
        The distance to the nearest obstacle from the given point.
    '''
    dist = 0
    while True:
        dist += 1
        for i in range(-dist, dist+1):
            for j in range(-dist, dist+1):
                x, y = point[0] + i, point[1] + j
                if x >= 5 and x < map.shape[0]-2 and y >= 0 and y < map.shape[1] and map[x, y] != 255:
                    return 1
                if dist > 5:
                    return 0

def A_star(map: np.ndarray, start: tuple, end: tuple) -> list:
    '''
    Finds the shortest path from the start to the end point using the A* algorithm.
    
    Args:
        map: The map of the maze.
        start: The start point.
        end: The end point.

    Returns:
        A list of points representing the shortest path from the start to the end point.
    '''
    # Priority queue to store (cost, current_node, path)
    priority_queue = [(0, end, [])]
    visited = set()

    while priority_queue:
        current_cost, current_node, path = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue

        visited.add(current_node)
        path = path + [current_node]
        
        if current_node == start:
            return path
        
        for neighbor in neighbors(map, current_node):
            if neighbor not in visited:
                # Add the heuristic cost to the priority queue
                heapq.heappush(priority_queue, (current_cost + 1 + np.linalg.norm(np.array(neighbor) - np.array(start)), neighbor, path))
    
    return None  # If no path exists

def draw_path(map: np.ndarray, path: list) -> np.ndarray:
    '''
    Draws the path on the map.
    
    Args:
        map: The map of the maze.
    '''
    for point in path:
        map[point[0], point[1]] = 100
    
    return map

def draw_color_path(map: np.ndarray, path: list) -> np.ndarray:
    '''
    Draws the path on the map.
    
    Args:
        map: The map of the maze.
    '''
    cv2.circle(map, (start[1], start[0]), 2, (255, 0, 0), -1)
    cv2.circle(map, (end[1], end[0]), 2, (0, 0, 255), -1)
    for point in path:
        map[point[0], point[1]] = [0, 255, 0]
    
    return map

if __name__ == "__main__":
    
    # Load the image
    image = cv2.imread('maze.png')

    start = (5, 195)
    end = (405, 215)

    # Convert the image to grayscale
    map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # Find the shortest path using A* algorithm
    path = A_star(map, start, end)
    print(path)

    # Draw the path on the map
    # map = draw_path(map, path)
    image = draw_color_path(image, path)

    # cv2.imshow('Map', map)
    cv2.imshow('Colored', image)
    cv2.imwrite('maze_solved.png', image)
    cv2.waitKey(0)
    
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    