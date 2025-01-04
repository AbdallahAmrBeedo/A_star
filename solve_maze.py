import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

def load_map(file_path: str) -> np.ndarray:
    '''
    Loads the map of the maze from the given file.
    
    Args:
        file_path: The path to the file containing the map.

    Returns:
        Original image of the maze.
        The map of the maze.
    '''
    # Load the image from the file
    image = cv2.imread(file_path)
    # Convert the image to grayscale
    map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, map

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
            if x >= 5 and x < map.shape[0]-2 and y >= 0 and y < map.shape[1] and map[x, y] == 255: 
                # if nearest_obs_2(map, (x, y)) >= 5:
                    neighbors.append((x, y))
    return neighbors

def nearest_obs(map: np.ndarray, point: tuple) -> int:
    '''
    Returns the distance to the nearest obstacle from the given point.

    Args:
        map: The map of the maze.
        point: The point to find the distance to the nearest obstacle from.

    Returns:
        int: The distance to the nearest obstacle from the given point.
    '''
    dist = 1
    while True:
        try:
            # Sum of cells in the side of the square of size 2*dist + 1
            sum_cells = np.sum(map[point[0]-dist, point[1]-dist:point[1]+dist+1]) + np.sum(map[point[0]+dist, point[1]-dist:point[1]+dist+1]) + np.sum(map[point[0]-dist+1:point[0]+dist, point[1]-dist]) + np.sum(map[point[0]-dist+1:point[0]+dist, point[1]+dist])
        except IndexError:
            dist += 1
            if dist >= 9:
                return 10
            continue
        num_cells = np.ceil((2*dist +1) **2 - (2*dist - 1) **2)
        # print(sum_cells, num_cells)
        if np.ceil(sum_cells / 255) < num_cells:
            return dist
        if dist >= 9:
            return 10
        dist += 1

def A_star(map: np.ndarray, start: tuple, end: tuple, safe_radius: int) -> list:
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
                if safe_radius !=0:
                    if nearest_obs(map, neighbor) < safe_radius:
                        continue
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
    cv2.circle(map, (path[0][1], path[0][0]), 2, (255, 0, 0), -1)
    cv2.circle(map, (path[-1][1], path[-1][0]), 2, (0, 0, 255), -1)
    for point in path:
        map[point[0], point[1]] = [0, 255, 0]
        cv2.imshow('Final Path', map)
        cv2.waitKey(1)
    
    return map

def main():

    start = (5, 195)
    end = (405, 215)

    # Radius of the circle around the point to check for obstacles
    # 0 means no check and if more than 0, the value should be less than 8
    safe_radius = 7
    # Load the map of the maze
    image, map = load_map('Maze_img/maze.png')

    # Find the shortest path using A* algorithm
    path = A_star(map, start, end, safe_radius)

    if path is None:
        raise ValueError('No path exists from the start to the end point.')
    
    # Draw the path on the map
    image = draw_color_path(image, path)

    cv2.imwrite('maze_solved.png', image)
    cv2.imshow('Final Path', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()